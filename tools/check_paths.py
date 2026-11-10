from pathlib import Path
import re

repo = Path("meridian-workshop")
bad = []
for p in repo.rglob("*.py"):
    txt = p.read_text(encoding="utf-8")
    for i, line in enumerate(txt.splitlines(), 1):
        if re.search(r'\+\s*["\']/', line) or re.search(r'["\']/["\']\s*\+', line):
            bad.append((str(p), i, line))

print(f"Hardcoded slash path joins found: {len(bad)}")
for b in bad:
    print(b)
