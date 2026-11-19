"""Generate a compact, validated single-file AI research review via a round adapter."""
import argparse
import csv
import hashlib
import importlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MISSING = 'N/A'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def table(headers, rows):
    def line(items):
        return '| ' + ' | '.join(str(item).replace('|', r'\|') for item in items) + ' |'
    return '\n'.join([line(headers), line(['---']*len(headers)), *(line(row) for row in rows)])


def csv_block(name, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: MISSING if value is None else value for key, value in row.items()} for row in rows)
    return f'### {name}\n\n```csv\n{stream.getvalue()}```'


def csv_tables(document):
    return [list(csv.DictReader(io.StringIO(block))) for block in document.split('```csv\n')[1:]
            for block in [block.split('\n```', 1)[0]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='EXP007', help='Experiment ID (adapter scripts.review_expNNN)')
    parser.add_argument('--out', type=Path, default=Path('dist/AI_RESEARCH_REVIEW.md'))
    parser.add_argument('--verify', action='store_true', help='Compare an existing review with the current verified export')
    args = parser.parse_args()
    experiment = args.experiment.upper()
    if len(experiment) != 6 or not experiment.startswith('EXP') or not experiment[3:].isascii() or not experiment[3:].isdigit():
        parser.error('Expected EXP followed by three ASCII digits')
    adapter = importlib.import_module(f'scripts.review_{experiment.lower()}')
    document = adapter.build()
    encoded = document.encode('utf-8')
    if args.verify:
        if args.out.read_bytes() != encoded:
            raise ValueError('Review differs from validated evidence; regenerate it before submission')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(encoded)
        if args.out.read_text(encoding='utf-8') != document:
            raise AssertionError('Review UTF-8 round-trip failed')
    print(f'{args.out}: {len(encoded):,} UTF-8 bytes; evidence validation passed')


if __name__ == '__main__':
    main()
