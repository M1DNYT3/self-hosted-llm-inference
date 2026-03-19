#!/usr/bin/env python3
"""
sanitize.py — No-op passthrough.

Email redaction is now handled at export time inside generate_dump.py via
PostgreSQL regexp_replace, so the raw dump is already clean when written.

This file is kept for reference and as a pipeline hook if additional
post-export filtering is ever needed.

Usage (still works as before, but does nothing):
    python fixture/sanitize.py < fixture/dump_raw.sql | gzip -9 > fixture/dump.sql.gz
"""

import sys


def sanitize(text: str) -> str:
    return text


if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(sanitize(line))
