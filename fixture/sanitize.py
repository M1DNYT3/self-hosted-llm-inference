#!/usr/bin/env python3
"""
sanitize.py — Strip PII from the fixture dump before committing.

Run AFTER exporting the raw dump, BEFORE gzipping and committing:
    python fixture/sanitize.py fixture/dump_raw.sql > fixture/dump_clean.sql
    gzip -9 -c fixture/dump_clean.sql > fixture/dump.sql.gz

What it removes:
  - Email addresses (regex)
  - Phone numbers (regex)
  - Replaces with [REDACTED]

What it does NOT touch:
  - Job descriptions (needed for inference tasks)
  - Company names (needed for matching)
  - Score values (needed for company_enrich task)
"""

import re
import sys

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(\+?\d[\d\s\-().]{7,}\d)(?!\d)",
    re.IGNORECASE,
)


def sanitize(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(sanitize(line))
