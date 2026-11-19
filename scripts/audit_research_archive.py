"""Check that historical evidence survived relocation byte-for-byte."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.verify_package import ROOT, _payloads


def audit():
    payload = _payloads()
    by_hash = {}
    for path in payload:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash.setdefault(digest, []).append(path.relative_to(ROOT).as_posix())
    previous = ROOT/'research/history/handover_snapshots/PRIOR_RELEASE_SHA256.json'
    entries = json.loads(previous.read_text(encoding='utf8'))['files']
    fixed = ('checks/', 'reference/', 'evidence/', 'docs/')
    mapped = {}
    for entry in entries:
        source = entry['path']
        if source == 'README.md' or source.startswith(fixed):
            matches = sorted(by_hash.get(entry['sha256'], ()))
            choices = [path for path in matches if Path(path).name == Path(source).name]
            if not choices:
                raise ValueError(f'Missing/changed historical document or evidence: {source}')
            mapped[source] = {'sha256': entry['sha256'], 'paths': choices}
    current_docs = [
        ROOT/'README.md', *(ROOT/'docs').glob('*.md'), ROOT/'research/INDEX.md',
        ROOT/'research/CURRENT_BASELINE.md', ROOT/'research/history/INDEX.md',
        ROOT/'research/history/handover_snapshots/INDEX.md', *(ROOT/'research/history').glob('*/SUMMARY.md'),
        *(ROOT/'research/experiments').glob('*/README.md'), ROOT/'evidence/INDEX.md',
        *(ROOT.glob('SPIKE_*.md')), *(ROOT.glob('REWARD_RAW_INDEX.md')),
        *(ROOT/'research/experiments/EXP005_reward_spikes').glob('*.md'),
        *(ROOT/'research/experiments/EXP006_generalization').glob('*.md'),
        *(ROOT/'research/experiments/EXP007_macro_credit').glob('*.md'),
        *(ROOT/'research/experiments/EXP008_agent_d').glob('*.md'),
    ]
    for document in current_docs:
        for relative in re.findall(r'\]\(([^()\s#]+)\)', document.read_text(encoding='utf8')):
            if '://' not in relative and not (document.parent/relative).exists():
                raise ValueError(f'Broken current navigation: {document.relative_to(ROOT)} -> {relative}')
    print(f'Preserved {len(mapped)} original documents/evidence files; current navigation links passed')
    return mapped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-map', type=Path)
    args = parser.parse_args()
    result = audit()
    if args.write_map:
        args.write_map.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8', newline='\n')
