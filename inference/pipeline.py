# inference/pipeline.py
"""LLM refinement pipeline — producer/consumer/aggregator batch runner.

Entry points:
  run_batch(backend, task, limit, db_url)  — batch enrich DB records
  load_backend(task)                        — instantiate the configured backend

Lifecycle contract:
  backend.startup()  is always called before the batch
  backend.shutdown() is always called in finally — even if the batch errors out

Parallelism:
  backend.n_parallel (populated by startup()) controls the ThreadPoolExecutor
  pool size. All records are loaded before any worker starts — each record is
  processed by exactly one worker, no overlap.

Database:
  Uses raw SQLAlchemy text() queries against the fixture DB schema.
  All tables are in the 'market' schema.
"""

from __future__ import annotations

import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Engine

    from inference.base import BaseLLMBackend

_REANALYZE_DAYS = 30


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_batch(
    backend: "BaseLLMBackend",
    task: str,
    limit: int,
    db_url: str,
    force: bool = False,
) -> list[dict]:
    """Run a batch of LLM inference for the given task.

    Lifecycle: startup → batch → shutdown (always via finally).

    Args:
        backend:  Pre-instantiated backend (cpu/local/remote).
        task:     "job_skills" | "jd_reparse" | "jd_validate" | "company_enrich"
        limit:    Maximum records to process.
        db_url:   PostgreSQL connection URL (e.g. postgresql://user:pw@host/db).
        force:    Re-process already-enriched records.

    Returns list of per-record result dicts for downstream metrics collection.
    """
    engine = sa.create_engine(db_url, pool_size=backend.n_parallel + 1)
    try:
        backend.startup(queue_size=_count_pending(engine, task, limit, force))
        # n_parallel may be updated by startup() (Vast.ai computes from VRAM)
        engine.dispose()
        engine = sa.create_engine(db_url, pool_size=backend.n_parallel + 1)
        return _dispatch(backend, task, limit, force, engine)
    finally:
        backend.shutdown()
        engine.dispose()


def load_backend(task: str = "") -> "BaseLLMBackend":
    """Instantiate the configured GPU backend from env vars."""
    import inference.backends  # noqa: F401 — triggers @register() side-effects
    from inference.config import (
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_CONTEXT_WINDOW_COMPANY_ENRICH,
        LLM_CONTEXT_WINDOW_JD_REPARSE,
        LLM_CONTEXT_WINDOW_JOB_SKILLS,
        LLM_GPU_BACKEND,
        LLM_MODEL,
        LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH,
        LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE,
        LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE,
        LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS,
    )
    from inference.registry import get_backend

    kwargs: dict[str, object] = dict(
        base_url=LLM_BASE_URL, model=LLM_MODEL, api_key=LLM_API_KEY
    )

    if LLM_GPU_BACKEND == "remote":
        if task == "job_skills":
            kwargs["avg_inference_secs"] = LLM_VAST_AVG_INFERENCE_SECS_JOB_SKILLS
            kwargs["ctx_per_slot"] = LLM_CONTEXT_WINDOW_JOB_SKILLS
        elif task == "jd_reparse":
            kwargs["avg_inference_secs"] = LLM_VAST_AVG_INFERENCE_SECS_JD_REPARSE
            kwargs["ctx_per_slot"] = LLM_CONTEXT_WINDOW_JD_REPARSE
        elif task == "jd_validate":
            kwargs["avg_inference_secs"] = LLM_VAST_AVG_INFERENCE_SECS_JD_VALIDATE
            kwargs["ctx_per_slot"] = LLM_CONTEXT_WINDOW_JD_REPARSE
        else:
            kwargs["avg_inference_secs"] = LLM_VAST_AVG_INFERENCE_SECS_COMPANY_ENRICH
            kwargs["ctx_per_slot"] = LLM_CONTEXT_WINDOW_COMPANY_ENRICH

    return get_backend(LLM_GPU_BACKEND, **kwargs)


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    backend: "BaseLLMBackend",
    task: str,
    limit: int,
    force: bool,
    engine: "Engine",
) -> list[dict]:
    with engine.connect() as conn:
        if task == "job_skills":
            return _run_job_skills_batch(conn, backend, limit, force)
        elif task == "jd_reparse":
            return _run_jd_reparse_batch(conn, backend, limit, force)
        elif task == "jd_validate":
            return _run_jd_validate_batch(conn, backend, limit, force)
        elif task == "company_enrich":
            return _run_company_enrich_batch(conn, backend, limit, force)
        else:
            raise ValueError(
                f"Unknown task: '{task}'. "
                "Choose: job_skills, jd_reparse, jd_validate, company_enrich"
            )


def _count_pending(
    engine: "Engine", task: str, limit: int, force: bool
) -> int:
    with engine.connect() as conn:
        if task == "job_skills":
            return _count_pending_job_skills(conn, limit, force)
        elif task == "jd_reparse":
            return _count_pending_jd_reparse(conn, limit, force)
        elif task == "jd_validate":
            return _count_pending_jd_validate(conn, limit, force)
        elif task == "company_enrich":
            return _count_pending_company_enrich(conn, limit, force)
        return 0


def _parse_llm_json(content: str) -> tuple[dict | None, bool]:
    """Parse LLM response content as JSON dict. Returns (parsed, success)."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed, True
    except (json.JSONDecodeError, ValueError):
        pass
    return None, False


def _fmt_elapsed(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{int(secs // 60)}m{int(secs % 60):02d}s"


# ---------------------------------------------------------------------------
# job_skills batch
# ---------------------------------------------------------------------------
# Fixture DB schema:
#   market.jobs_raw    (id, title, description)
#   market.jobs_derived (id, job_fk, section_skills, skill_tokens, ...)
#   market.job_skills_premium (job_fk, required jsonb, preferred jsonb,
#                              enriched_at, reanalyze_due_at)
# ---------------------------------------------------------------------------


def _count_pending_job_skills(
    conn: "Connection", limit: int, force: bool
) -> int:
    if force:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_raw "
                "WHERE description IS NOT NULL"
            )
        )
    else:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_raw r "
                "WHERE r.description IS NOT NULL "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM market.job_skills_premium p "
                "  WHERE p.job_fk = r.id"
                ")"
            )
        )
    row = result.fetchone()
    return min(int(row[0]) if row else 0, limit)


def _job_skills_worker(
    backend: "BaseLLMBackend",
    work_q: queue.Queue,
    result_q: queue.Queue,
    worker_idx: int,
    n_workers: int,
) -> None:
    """Pull jobs from work_q, call LLM, push results to result_q."""
    from inference.base import LLMRequest
    from inference.config import LLM_CONTEXT_WINDOW_JOB_SKILLS, LLM_MAX_TOKENS
    from inference.prompts import build_job_skills_prompt

    prefix = f"[w{worker_idx}/{n_workers}] " if n_workers > 1 else ""

    while True:
        try:
            row = work_q.get_nowait()
        except queue.Empty:
            break

        job_id, title, description, section_skills, skill_tokens = row
        try:
            system_prompt, user_prompt = build_job_skills_prompt(
                title,
                description,
                section_skills=section_skills,
                skill_tokens=skill_tokens,
                context_window=LLM_CONTEXT_WINDOW_JOB_SKILLS,
            )
            request = LLMRequest(
                task="job_skills",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=LLM_MAX_TOKENS,
            )
            response = backend.complete(request)

            parsed, parse_success = _parse_llm_json(response.content)
            response.parse_success = parse_success
            response.parsed = parsed

            status = "ok" if parse_success else "parse_fail"
            lbl = f"[{response.worker_label}] " if response.worker_label else ""
            print(
                f"  {lbl}{prefix}job {job_id:>6}  {response.latency_ms:>5}ms  "
                f"in={response.input_tokens} out={response.output_tokens}  [{status}]",
                flush=True,
            )

            if parse_success:
                result_q.put(
                    {
                        "job_id": job_id,
                        "required": (parsed or {}).get("required"),
                        "preferred": (parsed or {}).get("preferred"),
                        "latency_ms": response.latency_ms,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "status": status,
                    }
                )
            else:
                result_q.put({"job_id": job_id, "status": "parse_fail"})

        except Exception as exc:
            print(f"  {prefix}job {job_id:>6}  ERROR: {exc}", flush=True)
            result_q.put({"job_id": job_id, "status": "failed"})
        finally:
            work_q.task_done()


def _run_job_skills_batch(
    conn: "Connection",
    backend: "BaseLLMBackend",
    limit: int,
    force: bool,
) -> list[dict]:
    """Extract required/preferred skills from job postings.

    Producer/Consumer/Aggregator pattern:
      - Main thread loads all records → work_q
      - ThreadPoolExecutor workers pull from work_q, call LLM → result_q
      - Main thread aggregator reads result_q, bulk-commits every 50 successes
    """
    if force:
        query = text(
            "SELECT r.id, r.title, r.description, "
            "       d.section_skills, d.skill_tokens "
            "FROM market.jobs_raw r "
            "LEFT JOIN market.jobs_derived d ON d.job_fk = r.id "
            "WHERE r.description IS NOT NULL "
            "LIMIT :limit"
        )
    else:
        query = text(
            "SELECT r.id, r.title, r.description, "
            "       d.section_skills, d.skill_tokens "
            "FROM market.jobs_raw r "
            "LEFT JOIN market.jobs_derived d ON d.job_fk = r.id "
            "WHERE r.description IS NOT NULL "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM market.job_skills_premium p WHERE p.job_fk = r.id"
            ") "
            "LIMIT :limit"
        )

    rows = conn.execute(query, {"limit": limit}).fetchall()
    n = backend.n_parallel
    total = len(rows)
    print(f"[job_skills] Processing {total} job(s) ... (workers={n})")

    if total == 0:
        return []

    work_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    for row in rows:
        work_q.put(row)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            pool.submit(
                _job_skills_worker, backend, work_q, result_q, i + 1, n
            )

        summaries: list[dict] = []
        buffer: list[dict] = []
        processed_count = 0
        now = datetime.now(UTC)

        while processed_count < total:
            res = result_q.get()
            processed_count += 1
            summaries.append(res)

            if res["status"] == "ok":
                buffer.append(res)

            if len(buffer) >= 50:
                _persist_job_skills(conn, buffer, now)
                buffer = []

        if buffer:
            _persist_job_skills(conn, buffer, now)

    elapsed = time.monotonic() - t0
    ok = sum(1 for s in summaries if s["status"] == "ok")
    rate = len(summaries) / (elapsed / 60) if elapsed > 0 else 0
    print(
        f"[job_skills] Done. {ok}/{len(summaries)} parsed  "
        f"time={_fmt_elapsed(elapsed)}  rate={rate:.1f} rec/min"
    )
    return summaries


def _persist_job_skills(
    conn: "Connection", results: list[dict], now: datetime
) -> None:
    """Upsert a batch of job_skills results (INSERT ... ON CONFLICT DO UPDATE)."""
    reanalyze_at = now + timedelta(days=_REANALYZE_DAYS)
    conn.execute(
        text(
            "INSERT INTO market.job_skills_premium "
            "  (job_fk, required, preferred, enriched_at, reanalyze_due_at) "
            "VALUES (:job_fk, :required::jsonb, :preferred::jsonb, :enriched_at, :reanalyze_due_at) "
            "ON CONFLICT (job_fk) DO UPDATE SET "
            "  required = EXCLUDED.required, "
            "  preferred = EXCLUDED.preferred, "
            "  enriched_at = EXCLUDED.enriched_at, "
            "  reanalyze_due_at = EXCLUDED.reanalyze_due_at"
        ),
        [
            {
                "job_fk": r["job_id"],
                "required": json.dumps(r.get("required") or []),
                "preferred": json.dumps(r.get("preferred") or []),
                "enriched_at": now,
                "reanalyze_due_at": reanalyze_at,
            }
            for r in results
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# jd_reparse batch — slice-and-scan
# ---------------------------------------------------------------------------
# Schema: market.jobs_derived (id, job_fk, section_skills, section_about,
#   section_salary, section_job_type, section_contract, llm_reparsed_at)
# ---------------------------------------------------------------------------


def _count_pending_jd_reparse(
    conn: "Connection", limit: int, force: bool
) -> int:
    if force:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_derived d "
                "JOIN market.jobs_raw r ON r.id = d.job_fk "
                "WHERE r.description IS NOT NULL"
            )
        )
    else:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_derived d "
                "JOIN market.jobs_raw r ON r.id = d.job_fk "
                "WHERE r.description IS NOT NULL "
                "AND d.section_skills IS NULL "
                "AND d.llm_reparsed_at IS NULL"
            )
        )
    row = result.fetchone()
    return min(int(row[0]) if row else 0, limit)


def _jd_reparse_worker(
    backend: "BaseLLMBackend",
    work_q: queue.Queue,
    result_q: queue.Queue,
    worker_idx: int,
    n_workers: int,
) -> None:
    """Pull jobs from work_q, slice description, call LLM per slice, push merged result."""
    from inference.base import LLMRequest
    from inference.config import LLM_CONTEXT_WINDOW_JD_REPARSE, LLM_MAX_TOKENS
    from inference.prompts import (
        _merge_reparse_results,
        _outside_in_order,
        _slice_description,
        build_jd_reparse_prompt,
    )

    prefix = f"[w{worker_idx}/{n_workers}] " if n_workers > 1 else ""
    max_chars = int(LLM_CONTEXT_WINDOW_JD_REPARSE * 0.75) - 400

    while True:
        try:
            row = work_q.get_nowait()
        except queue.Empty:
            break

        derived_id, job_fk, title, description = row
        try:
            slices = _slice_description(description or "", max_chars)
            order = _outside_in_order(len(slices))

            slice_results: list[dict] = []
            total_latency_ms = 0
            total_in_tokens = 0
            total_out_tokens = 0
            worker_lbl = ""
            accumulated: dict[str, str] = {
                "section_about": "",
                "section_skills": "",
                "section_salary": "",
                "section_job_type": "",
                "section_contract": "",
            }

            for idx in order:
                excerpt = slices[idx]
                system_prompt, user_prompt = build_jd_reparse_prompt(title, excerpt)
                request = LLMRequest(
                    task="jd_reparse",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=LLM_MAX_TOKENS,
                )
                response = backend.complete(request)
                if not worker_lbl:
                    worker_lbl = response.worker_label
                total_latency_ms += response.latency_ms
                total_in_tokens += response.input_tokens
                total_out_tokens += response.output_tokens

                parsed, ok = _parse_llm_json(response.content)
                if ok and parsed:
                    slice_results.append(parsed)
                    for field in accumulated:
                        if not accumulated[field] and parsed.get(field):
                            accumulated[field] = parsed[field]
                    # Early-stop: all five sections found
                    if all(accumulated.values()):
                        break

            got_skills = bool(accumulated["section_skills"])
            n_slices = len(slice_results)
            lbl = f"[{worker_lbl}] " if worker_lbl else ""
            print(
                f"  {lbl}{prefix}job {job_fk:>6}  {total_latency_ms:>5}ms  "
                f"in={total_in_tokens} out={total_out_tokens}  "
                f"slices={n_slices}/{len(slices)}  "
                f"skills={'y' if got_skills else 'n'}  [ok]",
                flush=True,
            )

            result_q.put(
                {
                    "derived_id": derived_id,
                    "job_fk": job_fk,
                    "slice_results": slice_results,
                    "merged": _merge_reparse_results(slice_results),
                    "got_skills": got_skills,
                    "latency_ms": total_latency_ms,
                    "input_tokens": total_in_tokens,
                    "output_tokens": total_out_tokens,
                    "status": "ok",
                }
            )

        except Exception as exc:
            print(f"  {prefix}job {job_fk:>6}  ERROR: {exc}", flush=True)
            result_q.put({"job_fk": job_fk, "status": "failed"})
        finally:
            work_q.task_done()


def _run_jd_reparse_batch(
    conn: "Connection",
    backend: "BaseLLMBackend",
    limit: int,
    force: bool,
) -> list[dict]:
    """Re-parse job descriptions to fill missing section_* fields via slice-and-scan."""
    if force:
        query = text(
            "SELECT d.id, d.job_fk, r.title, r.description "
            "FROM market.jobs_derived d "
            "JOIN market.jobs_raw r ON r.id = d.job_fk "
            "WHERE r.description IS NOT NULL "
            "LIMIT :limit"
        )
    else:
        query = text(
            "SELECT d.id, d.job_fk, r.title, r.description "
            "FROM market.jobs_derived d "
            "JOIN market.jobs_raw r ON r.id = d.job_fk "
            "WHERE r.description IS NOT NULL "
            "AND d.section_skills IS NULL "
            "AND d.llm_reparsed_at IS NULL "
            "LIMIT :limit"
        )

    rows = conn.execute(query, {"limit": limit}).fetchall()
    n = backend.n_parallel
    total = len(rows)
    print(f"[jd_reparse] Processing {total} job(s) ... (workers={n})")

    if total == 0:
        return []

    work_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    for row in rows:
        work_q.put(row)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            pool.submit(
                _jd_reparse_worker, backend, work_q, result_q, i + 1, n
            )

        summaries: list[dict] = []
        buffer: list[dict] = []
        processed_count = 0
        now = datetime.now(UTC)

        while processed_count < total:
            res = result_q.get()
            processed_count += 1
            summaries.append(res)

            if res["status"] == "ok":
                buffer.append(res)

            if len(buffer) >= 50:
                _persist_jd_reparse(conn, buffer, now)
                buffer = []

        if buffer:
            _persist_jd_reparse(conn, buffer, now)

    elapsed = time.monotonic() - t0
    ok = sum(1 for s in summaries if s["status"] == "ok")
    got_skills = sum(1 for s in summaries if s.get("got_skills"))
    rate = len(summaries) / (elapsed / 60) if elapsed > 0 else 0
    print(
        f"[jd_reparse] Done. {ok}/{len(summaries)} processed  "
        f"skills_found={got_skills}  "
        f"time={_fmt_elapsed(elapsed)}  rate={rate:.1f} rec/min"
    )
    return summaries


def _persist_jd_reparse(
    conn: "Connection", results: list[dict], now: datetime
) -> None:
    """Update jd_reparse results: only write back fields that are currently null."""
    for res in results:
        merged = res["merged"]
        conn.execute(
            text(
                "UPDATE market.jobs_derived SET "
                "  section_skills    = COALESCE(section_skills,    :section_skills), "
                "  section_about     = COALESCE(section_about,     :section_about), "
                "  section_salary    = COALESCE(section_salary,    :section_salary), "
                "  section_job_type  = COALESCE(section_job_type,  :section_job_type), "
                "  section_contract  = COALESCE(section_contract,  :section_contract), "
                "  llm_reparsed_at   = :now "
                "WHERE id = :id"
            ),
            {
                "id": res["derived_id"],
                "section_skills": merged["section_skills"] or None,
                "section_about": merged["section_about"] or None,
                "section_salary": merged["section_salary"] or None,
                "section_job_type": merged["section_job_type"] or None,
                "section_contract": merged["section_contract"] or None,
                "now": now,
            },
        )
    conn.commit()


# ---------------------------------------------------------------------------
# jd_validate batch
# ---------------------------------------------------------------------------

_VALIDATE_SECTION_FIELDS = (
    "section_about",
    "section_skills",
    "section_salary",
    "section_job_type",
    "section_contract",
)


def _count_pending_jd_validate(
    conn: "Connection", limit: int, force: bool
) -> int:
    if force:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_derived "
                "WHERE section_skills IS NOT NULL"
            )
        )
    else:
        result = conn.execute(
            text(
                "SELECT COUNT(*) FROM market.jobs_derived "
                "WHERE section_skills IS NOT NULL "
                "AND llm_reparsed_at IS NULL"
            )
        )
    row = result.fetchone()
    return min(int(row[0]) if row else 0, limit)


def _jd_validate_worker(
    backend: "BaseLLMBackend",
    work_q: queue.Queue,
    result_q: queue.Queue,
    worker_idx: int,
    n_workers: int,
) -> None:
    """Validate heuristically-extracted sections, push results to result_q."""
    from inference.base import LLMRequest
    from inference.config import LLM_MAX_TOKENS
    from inference.prompts import build_jd_validate_prompt

    prefix = f"[w{worker_idx}/{n_workers}] " if n_workers > 1 else ""

    while True:
        try:
            row_tuple = work_q.get_nowait()
        except queue.Empty:
            break

        derived_id = row_tuple[0]
        job_fk = row_tuple[1]
        raw = dict(zip(_VALIDATE_SECTION_FIELDS, row_tuple[2:]))
        sections = {k: v for k, v in raw.items() if v}

        try:
            if not sections:
                result_q.put(
                    {
                        "derived_id": derived_id,
                        "job_fk": job_fk,
                        "status": "ok",
                        "cleared": 0,
                        "valid_sections": {},
                        "original_sections": sections,
                    }
                )
                continue

            system_prompt, user_prompt = build_jd_validate_prompt(sections)
            request = LLMRequest(
                task="jd_validate",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=LLM_MAX_TOKENS,
            )
            response = backend.complete(request)

            try:
                valid: dict = json.loads(response.content.strip())
            except json.JSONDecodeError:
                valid = {}

            cleared_count = sum(
                1
                for f in _VALIDATE_SECTION_FIELDS
                if f in sections and f not in valid
            )
            lbl = f"[{response.worker_label}] " if response.worker_label else ""
            cleared_short = [
                f.replace("section_", "")
                for f in _VALIDATE_SECTION_FIELDS
                if f in sections and f not in valid
            ]
            print(
                f"  {lbl}{prefix}job {job_fk:>6}  {response.latency_ms:>4}ms  "
                f"cleared={cleared_short or 'none'}  [ok]",
                flush=True,
            )

            result_q.put(
                {
                    "derived_id": derived_id,
                    "job_fk": job_fk,
                    "status": "ok",
                    "cleared": cleared_count,
                    "latency_ms": response.latency_ms,
                    "valid_sections": valid,
                    "original_sections": sections,
                }
            )

        except Exception as exc:
            print(f"  {prefix}job {job_fk:>6}  ERROR: {exc}", flush=True)
            result_q.put({"job_fk": job_fk, "status": "failed"})
        finally:
            work_q.task_done()


def _run_jd_validate_batch(
    conn: "Connection",
    backend: "BaseLLMBackend",
    limit: int,
    force: bool,
) -> list[dict]:
    """Validate all heuristically-extracted sections with JSON-in/JSON-out LLM call."""
    if force:
        query = text(
            "SELECT id, job_fk, section_about, section_skills, "
            "       section_salary, section_job_type, section_contract "
            "FROM market.jobs_derived "
            "WHERE section_skills IS NOT NULL "
            "LIMIT :limit"
        )
    else:
        query = text(
            "SELECT id, job_fk, section_about, section_skills, "
            "       section_salary, section_job_type, section_contract "
            "FROM market.jobs_derived "
            "WHERE section_skills IS NOT NULL "
            "AND llm_reparsed_at IS NULL "
            "LIMIT :limit"
        )

    rows = conn.execute(query, {"limit": limit}).fetchall()
    n = backend.n_parallel
    total = len(rows)
    print(f"[jd_validate] Validating {total} heuristic records ... (workers={n})")

    if total == 0:
        return []

    work_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    for row in rows:
        work_q.put(row)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            pool.submit(
                _jd_validate_worker, backend, work_q, result_q, i + 1, n
            )

        summaries: list[dict] = []
        buffer: list[dict] = []
        processed_count = 0
        now = datetime.now(UTC)

        while processed_count < total:
            res = result_q.get()
            processed_count += 1
            summaries.append(res)

            if res["status"] == "ok":
                buffer.append(res)

            if len(buffer) >= 50:
                _persist_jd_validate(conn, buffer, now)
                buffer = []

        if buffer:
            _persist_jd_validate(conn, buffer, now)

    elapsed = time.monotonic() - t0
    ok = sum(1 for s in summaries if s["status"] == "ok")
    cleared_any = sum(1 for s in summaries if s.get("cleared", 0) > 0)
    rate = len(summaries) / (elapsed / 60) if elapsed > 0 else 0
    print(
        f"[jd_validate] Done. {ok}/{len(summaries)} processed  "
        f"sections_cleared={cleared_any}  "
        f"time={_fmt_elapsed(elapsed)}  rate={rate:.1f} rec/min"
    )
    return summaries


def _persist_jd_validate(
    conn: "Connection", results: list[dict], now: datetime
) -> None:
    """Clear invalid sections and stamp llm_reparsed_at if at least one survived."""
    for res in results:
        valid = res["valid_sections"]
        original = res["original_sections"]

        updates: dict[str, object] = {"id": res["derived_id"]}
        set_clauses = []

        for field in _VALIDATE_SECTION_FIELDS:
            if field in original and field not in valid:
                set_clauses.append(f"{field} = NULL")
                if field == "section_skills":
                    set_clauses.append("skill_tokens = NULL")

        cleared_count = sum(
            1 for f in _VALIDATE_SECTION_FIELDS
            if f in original and f not in valid
        )
        # Only stamp if at least one section survived
        if cleared_count < len(original):
            set_clauses.append("llm_reparsed_at = :now")
            updates["now"] = now

        if set_clauses:
            conn.execute(
                text(
                    f"UPDATE market.jobs_derived SET {', '.join(set_clauses)} "
                    "WHERE id = :id"
                ),
                updates,
            )
    conn.commit()


# ---------------------------------------------------------------------------
# company_enrich batch
# ---------------------------------------------------------------------------
# Schema: market.companies (id, name, domain, industry)
#         market.company_scores (company_id, role_domain, computed_at,
#                                trust_score, trust_label, hiring_score, hiring_label)
#         market.company_scores_premium (company_id, role_domain, llm_result jsonb,
#                                        enriched_at, reanalyze_due_at)
# ---------------------------------------------------------------------------


def _count_pending_company_enrich(
    conn: "Connection", limit: int, force: bool
) -> int:
    if force:
        result = conn.execute(
            text(
                "SELECT COUNT(DISTINCT c.id) FROM market.companies c "
                "JOIN market.company_scores cs ON cs.company_id = c.id"
            )
        )
    else:
        result = conn.execute(
            text(
                "SELECT COUNT(DISTINCT c.id) FROM market.companies c "
                "JOIN market.company_scores cs ON cs.company_id = c.id "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM market.company_scores_premium p "
                "  WHERE p.company_id = c.id "
                "  AND p.llm_result IS NOT NULL "
                "  AND p.reanalyze_due_at > NOW()"
                ")"
            )
        )
    row = result.fetchone()
    return min(int(row[0]) if row else 0, limit)


def _company_enrich_worker(
    backend: "BaseLLMBackend",
    work_q: queue.Queue,
    result_q: queue.Queue,
    worker_idx: int,
    n_workers: int,
) -> None:
    """Pull company payloads from work_q, call LLM, push results to result_q."""
    from inference.base import LLMRequest
    from inference.config import LLM_MAX_TOKENS
    from inference.prompts import build_company_enrich_prompt

    prefix = f"[w{worker_idx}/{n_workers}] " if n_workers > 1 else ""

    while True:
        try:
            company_id, company_name, payload = work_q.get_nowait()
        except queue.Empty:
            break

        try:
            system_prompt, user_prompt = build_company_enrich_prompt(payload)
            request = LLMRequest(
                task="company_enrich",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=LLM_MAX_TOKENS,
            )
            response = backend.complete(request)

            parsed, parse_success = _parse_llm_json(response.content)
            status = "ok" if parse_success else "parse_fail"
            lbl = f"[{response.worker_label}] " if response.worker_label else ""
            print(
                f"  {lbl}{prefix}{company_name:<30}  {response.latency_ms:>5}ms  "
                f"in={response.input_tokens} out={response.output_tokens}  [{status}]",
                flush=True,
            )

            result_q.put(
                {
                    "company_id": company_id,
                    "company_name": company_name,
                    "llm_result": parsed if parse_success else {"raw": response.content},
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "status": status,
                }
            )
        except Exception as exc:
            print(f"  {prefix}{company_name:<30}  ERROR: {exc}", flush=True)
            result_q.put({"company_name": company_name, "status": "failed"})
        finally:
            work_q.task_done()


def _run_company_enrich_batch(
    conn: "Connection",
    backend: "BaseLLMBackend",
    limit: int,
    force: bool,
) -> list[dict]:
    """Enrich company records with LLM-generated summary and signals."""
    if force:
        query = text(
            "SELECT DISTINCT ON (c.id) c.id, c.name, c.domain, c.industry, "
            "  cs.role_domain, cs.trust_score, cs.trust_label, "
            "  cs.hiring_score, cs.hiring_label "
            "FROM market.companies c "
            "JOIN market.company_scores cs ON cs.company_id = c.id "
            "LIMIT :limit"
        )
    else:
        query = text(
            "SELECT DISTINCT ON (c.id) c.id, c.name, c.domain, c.industry, "
            "  cs.role_domain, cs.trust_score, cs.trust_label, "
            "  cs.hiring_score, cs.hiring_label "
            "FROM market.companies c "
            "JOIN market.company_scores cs ON cs.company_id = c.id "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM market.company_scores_premium p "
            "  WHERE p.company_id = c.id "
            "  AND p.llm_result IS NOT NULL "
            "  AND p.reanalyze_due_at > NOW()"
            ") "
            "LIMIT :limit"
        )

    rows = conn.execute(query, {"limit": limit}).fetchall()
    n = backend.n_parallel
    total = len(rows)
    print(f"[company_enrich] Processing {total} company/ies ... (workers={n})")

    if total == 0:
        return []

    work_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    now = datetime.now(UTC)

    for row in rows:
        payload = {
            "company_name": row[1],
            "company_domain": row[2],
            "company_industry": row[3],
            "heuristic_scores": [
                {
                    "role_domain": row[4],
                    "trust_score": row[5],
                    "trust_label": row[6],
                    "hiring_score": row[7],
                    "hiring_label": row[8],
                }
            ],
        }
        work_q.put((row[0], row[1], payload))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        for i in range(n):
            pool.submit(
                _company_enrich_worker, backend, work_q, result_q, i + 1, n
            )

        summaries: list[dict] = []
        buffer: list[dict] = []
        processed_count = 0

        while processed_count < total:
            res = result_q.get()
            processed_count += 1
            summaries.append(res)

            if res["status"] == "ok":
                buffer.append(res)

            if len(buffer) >= 50:
                _persist_company_enrich(conn, buffer, now)
                buffer = []

        if buffer:
            _persist_company_enrich(conn, buffer, now)

    elapsed = time.monotonic() - t0
    ok = sum(1 for s in summaries if s["status"] == "ok")
    rate = len(summaries) / (elapsed / 60) if elapsed > 0 else 0
    print(
        f"[company_enrich] Done. {ok}/{len(summaries)} parsed  "
        f"time={_fmt_elapsed(elapsed)}  rate={rate:.1f} rec/min"
    )
    return summaries


def _persist_company_enrich(
    conn: "Connection", results: list[dict], now: datetime
) -> None:
    """Upsert company enrichment results."""
    reanalyze_at = now + timedelta(days=_REANALYZE_DAYS)
    conn.execute(
        text(
            "INSERT INTO market.company_scores_premium "
            "  (company_id, role_domain, llm_result, enriched_at, reanalyze_due_at) "
            "VALUES (:company_id, '__all__', :llm_result::jsonb, :enriched_at, :reanalyze_due_at) "
            "ON CONFLICT (company_id, role_domain) DO UPDATE SET "
            "  llm_result = EXCLUDED.llm_result, "
            "  enriched_at = EXCLUDED.enriched_at, "
            "  reanalyze_due_at = EXCLUDED.reanalyze_due_at"
        ),
        [
            {
                "company_id": r["company_id"],
                "llm_result": json.dumps(r["llm_result"]),
                "enriched_at": now,
                "reanalyze_due_at": reanalyze_at,
            }
            for r in results
        ],
    )
    conn.commit()
