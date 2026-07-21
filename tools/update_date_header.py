#!/usr/bin/env python3
# Last updated: 2026-07-21
"""Stamp/refresh a '# Last updated: YYYY-MM-DD' header on Python files.

Usage:
    python3 tools/update_date_header.py                 # scan whole repo (skips .venv, .git, __pycache__)
    python3 tools/update_date_header.py path/to/file.py  # stamp a single file
"""

import datetime
import os
import re
import sys

MARKER = "# Last updated:"
SKIP_DIRS = {".venv", ".git", "__pycache__", "venv", "node_modules"}


def stamp(file_path):
    if not file_path.endswith(".py") or not os.path.isfile(file_path):
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    today = datetime.date.today().isoformat()
    new_line = f"{MARKER} {today}\n"

    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and re.match(r"#.*coding[:=]", lines[insert_at]):
        insert_at += 1

    changed = False
    for i, line in enumerate(lines[: insert_at + 2]):
        if line.startswith(MARKER):
            if line != new_line:
                lines[i] = new_line
                changed = True
            break
    else:
        lines.insert(insert_at, new_line)
        changed = True

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return changed


def scan(root):
    stamped = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                if stamp(path):
                    stamped.append(path)
    return stamped


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stamp(sys.argv[1])
    else:
        changed = scan(".")
        for path in changed:
            print(f"stamped: {path}")
        if not changed:
            print("no files needed stamping")
