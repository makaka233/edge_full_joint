"""Verify exact SHA256 contents of the self-contained source handover."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'SHA256_MANIFEST.json'

def _payloads():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and
        p.name!='SHA256_MANIFEST.json' and
        '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and
        not p.suffix in ('.pt','.pth','.ckpt','.pyc') and
        '.venv' not in p.parts and 'runs' not in p.parts)

def verify():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    listed={item['path']:item for item in manifest['files']}
    found={str(p.relative_to(ROOT)):p for p in _payloads()}
    errors=[]
    for name,item in listed.items():
        if name not in found:errors.append((name,'missing'));continue
        data=found[name].read_bytes()
        if item['size']!=len(data) or item['sha256']!=hashlib.sha256(data).hexdigest():
            errors.append((name,'SHA256/size mismatch'))
    for name in found.keys()-listed.keys():errors.append((name,'untracked'))
    forbidden=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
               if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt')]
    if forbidden:errors.append(('weights files in package',forbidden))
    print(f"SHA256 payload: {len(listed)-len(errors)}/{len(listed)}; extra/missing/changed={len(errors)}")
    for row in errors:print('ERROR:',row)
    return not errors
if __name__=='__main__':sys.exit(0 if verify() else 1)
