"""Generate the weight-free research bundle and exact payload manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.verify_package import MANIFEST, ROOT, _payloads, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'dist'/'edge_joint_research.zip')
    args = parser.parse_args()
    target = args.out.resolve()
    if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT/'dist'):
        raise ValueError('Keep in-tree archives under dist/ to avoid recursively packaging results')
    payloads = _payloads()
    entries = [{'path': path.relative_to(ROOT).as_posix(), 'size': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in payloads]
    MANIFEST.write_text(json.dumps({'purpose': 'Standalone research source and evidence; no weights',
                                    'files': entries}, indent=2)+'\n', encoding='utf8')
    if not verify():
        raise RuntimeError('Source manifest verification failed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in [*payloads, MANIFEST]:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    print(f'{target} SHA256 {hashlib.sha256(target.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
