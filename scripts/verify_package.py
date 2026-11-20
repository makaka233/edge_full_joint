"""Verify exact SHA256 contents of the self-contained source handover."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'SHA256_MANIFEST.json'
EXCLUDED_DIRS={'.git','.venv','__pycache__','.pytest_cache','runs','dist'}
WEIGHT_SUFFIXES={'.pt','.pth','.ckpt'}

def _included_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and
                  not EXCLUDED_DIRS.intersection(p.relative_to(ROOT).parts))

def _payloads():
    return [p for p in _included_files() if p!=MANIFEST and
            p.suffix.lower() not in (WEIGHT_SUFFIXES | {'.pyc'})]

def verify():
    if not MANIFEST.is_file():
        print(f'Package manifest missing: {MANIFEST}. Run this check from an extracted delivery.')
        return False
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    listed={item['path']:item for item in manifest['files']}
    found={p.relative_to(ROOT).as_posix():p for p in _payloads()}
    errors=[]
    for name,item in listed.items():
        if name not in found:errors.append((name,'missing'));continue
        data=found[name].read_bytes()
        if item['size']!=len(data) or item['sha256']!=hashlib.sha256(data).hexdigest():
            errors.append((name,'SHA256/size mismatch'))
    for name in found.keys()-listed.keys():errors.append((name,'untracked'))
    forbidden=[p.relative_to(ROOT).as_posix() for p in _included_files()
               if p.suffix.lower() in WEIGHT_SUFFIXES]
    if forbidden:errors.append(('weights files in package',forbidden))
    print(f"SHA256 payload: {len(listed)-len(errors)}/{len(listed)}; extra/missing/changed={len(errors)}")
    for row in errors:print('ERROR:',row)
    return not errors
if __name__=='__main__':sys.exit(0 if verify() else 1)
