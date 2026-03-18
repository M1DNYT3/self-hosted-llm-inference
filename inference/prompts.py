# inference/prompts.py
"""Prompt builders and text utilities for the inference pipeline.

Each task's prompt builder returns (system_prompt, user_prompt).
The actual prompt text is intentionally generic here — the real system uses
domain-specific prompts for the job market. What matters for the case study
is the interface and the slice-and-scan infrastructure for jd_reparse.
"""

from typing import Any


# ---------------------------------------------------------------------------
# job_skills
# ---------------------------------------------------------------------------


def build_job_skills_prompt(
    title: str,
    description: str,
    section_skills: str | None = None,
    skill_tokens: str | None = None,
    context_window: int = 4096,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the job_skills task.

    Extracts required and preferred skills from a job posting.
    Output: {"required": [...], "preferred": [...]}

    The description is truncated to fit within the context window.
    section_skills / skill_tokens are heuristically pre-extracted hints
    that reduce the LLM's search space.
    """
    max_desc_chars = int(context_window * 0.75) - 400
    truncated = (description or "")[:max_desc_chars]

    system_prompt = (
        "You are a precise skill extractor. "
        "Given a job title and description, return a JSON object with two keys: "
        '"required" (a list of required skills) and "preferred" (a list of nice-to-have skills). '
        "Return only valid JSON, no markdown fences."
    )

    hints = ""
    if section_skills:
        hints += f"\n\nHeuristically extracted skills section:\n{section_skills}"
    if skill_tokens:
        hints += f"\n\nSkill tokens found: {skill_tokens}"

    user_prompt = (
        f"Job title: {title or 'N/A'}\n\n"
        f"Description:\n{truncated}"
        f"{hints}\n\n"
        'Return JSON: {"required": [...], "preferred": [...]}'
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# jd_reparse — slice-and-scan
# ---------------------------------------------------------------------------


def build_jd_reparse_prompt(
    title: str,
    excerpt: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one description slice.

    Each record may span multiple LLM calls (one per slice). The pipeline
    uses _outside_in_order() to scan slices starting from the most
    information-dense regions (end and beginning of long descriptions).

    Output: {
      "section_skills": "...",
      "section_about": "...",
      "section_salary": "...",
      "section_job_type": "...",
      "section_contract": "..."
    }
    Only include keys where content was found.
    """
    system_prompt = (
        "You are a job description parser. "
        "Given an excerpt from a job posting, extract any of the following sections "
        "if present: section_skills, section_about, section_salary, section_job_type, "
        "section_contract. "
        "Return only a JSON object with the keys you found. "
        "Omit keys for sections not present in this excerpt. "
        "Return raw JSON, no markdown."
    )
    user_prompt = (
        f"Job title: {title or 'N/A'}\n\n"
        f"Excerpt:\n{excerpt}\n\n"
        "Extract any sections present and return JSON."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# jd_validate
# ---------------------------------------------------------------------------


def build_jd_validate_prompt(
    sections: dict[str, str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the jd_validate task.

    Validates heuristically-extracted sections. Returns only the sections
    that appear genuine (not noise/boilerplate). Sections absent from the
    output are cleared in the DB.

    Output: subset of the input keys that are valid.
    """
    sections_text = "\n\n".join(
        f"[{k}]\n{v}" for k, v in sections.items() if v
    )
    system_prompt = (
        "You are a job description validator. "
        "Given extracted sections from a job posting, return a JSON object "
        "containing only the sections that appear genuine and relevant. "
        "Omit any section that looks like boilerplate, noise, or a mis-extraction. "
        "Return the same key names as the input. Raw JSON only."
    )
    user_prompt = (
        f"Extracted sections:\n\n{sections_text}\n\n"
        "Return JSON with only the valid sections."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# company_enrich
# ---------------------------------------------------------------------------


def build_company_enrich_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the company_enrich task.

    payload keys used: company_name, company_domain, company_industry,
    heuristic_scores (list of dicts with role_domain/trust_score/etc.)

    Output: {"summary": "...", "signals": [...]}
    """
    name = payload.get("company_name", "Unknown")
    domain = payload.get("company_domain") or "unknown"
    industry = payload.get("company_industry") or "unknown"

    scores_text = ""
    for s in payload.get("heuristic_scores", []):
        scores_text += (
            f"  [{s.get('role_domain')}] "
            f"trust={s.get('trust_score')} ({s.get('trust_label')}), "
            f"hiring={s.get('hiring_score')} ({s.get('hiring_label')})\n"
        )

    system_prompt = (
        "You are a company analyst. "
        "Given company metadata and heuristic scores, return a JSON object with: "
        '"summary" (1-2 sentence company description) and '
        '"signals" (list of notable hiring or trust signals). '
        "Raw JSON only."
    )
    user_prompt = (
        f"Company: {name}\n"
        f"Domain: {domain}\n"
        f"Industry: {industry}\n"
        f"Heuristic scores:\n{scores_text or '  (none)'}\n\n"
        'Return JSON: {"summary": "...", "signals": [...]}'
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Text utilities used by jd_reparse
# ---------------------------------------------------------------------------


def _slice_description(text: str, max_chars: int) -> list[str]:
    """Split text into paragraph-aligned chunks no larger than max_chars.

    Tries paragraph breaks (\\n\\n), then line breaks (\\n), then hard split.
    Never produces an empty slice list.
    """
    if len(text) <= max_chars:
        return [text]

    slices: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        chunk = remaining[:max_chars]
        cut = chunk.rfind("\n\n")
        if cut > max_chars // 4:
            slices.append(remaining[: cut + 2].rstrip())
            remaining = remaining[cut + 2 :].lstrip()
            continue
        cut = chunk.rfind("\n")
        if cut > max_chars // 4:
            slices.append(remaining[:cut].rstrip())
            remaining = remaining[cut + 1 :].lstrip()
            continue
        slices.append(chunk)
        remaining = remaining[max_chars:]

    if remaining.strip():
        slices.append(remaining.strip())
    return slices or [text]


def _outside_in_order(n: int) -> list[int]:
    """Return 0-indexed scan order: last, first, second-to-last, second, ...

    Rationale: job description sections are most commonly near the end
    (requirements) or beginning (overview). Scanning outside-in and stopping
    early when all sections are found reduces average LLM calls per record.

    Examples:
      n=1 → [0]
      n=4 → [3, 0, 2, 1]
      n=7 → [6, 0, 5, 1, 4, 2, 3]
    """
    order: list[int] = []
    lo, hi = 0, n - 1
    turn = 0
    while lo <= hi:
        if turn % 2 == 0:
            order.append(hi)
            hi -= 1
        else:
            order.append(lo)
            lo += 1
        turn += 1
    return order


def _merge_reparse_results(results: list[dict]) -> dict:
    """Merge slice results: first non-empty value wins for each field."""
    merged: dict = {
        "section_about": "",
        "section_skills": "",
        "section_salary": "",
        "section_job_type": "",
        "section_contract": "",
    }
    for r in results:
        for field in merged:
            if not merged[field] and r.get(field):
                merged[field] = r[field]
    return merged
