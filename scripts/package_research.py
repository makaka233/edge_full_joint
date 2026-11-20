"""Generate the weight-free research bundle and exact payload manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.verify_package import MANIFEST, ROOT, _payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='destination ZIP for an explicit package delivery')
    args = parser.parse_args()
    target = args.out.resolve()
    if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT/'runs'/'results'):
        raise ValueError('Keep in-tree archives under runs/results/<EXPERIMENT_ID>')
    payloads = _payloads()
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in payloads:
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            bundle.writestr(name, data)
            entries.append({'path': name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        manifest = {'purpose': 'Standalone research source; no weights', 'files': entries}
        bundle.writestr(MANIFEST.name, json.dumps(manifest, indent=2)+'\n')
    print(f'{target} SHA256 {hashlib.sha256(target.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
