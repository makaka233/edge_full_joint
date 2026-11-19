"""Atomic, streaming full-state checkpoint. Avoid torch.save's monolithic numpy pickle."""
import io,pickle,zipfile,torch
from pathlib import Path

def save_checkpoint(payload,path):
    path=Path(path);tmp=path.with_suffix('.tmp');core={k:v for k,v in payload.items() if k not in ('dr','sr','local_sr','aux_dr')}
    b=io.BytesIO();torch.save(core,b)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr('core.pt',b.getvalue())
        with z.open('replay.pkl','w',force_zip64=True) as f:pickle.dump((payload['dr'],payload['sr'],payload.get('local_sr'),payload.get('aux_dr')),f,protocol=5)
    tmp.replace(path)

def load_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        if 'core.pt' in z.namelist():
            x=torch.load(io.BytesIO(z.read('core.pt')),map_location='cpu',weights_only=False)
            with z.open('replay.pkl') as f:replays=pickle.load(f)
            x['dr'],x['sr']=replays[:2]
            if len(replays)>=3:x['local_sr']=replays[2]
            if len(replays)>=4:x['aux_dr']=replays[3]
            return x
    return torch.load(path,map_location='cpu',weights_only=False)

def read_core(path):
    with zipfile.ZipFile(path) as z:
        if 'core.pt' in z.namelist():return torch.load(io.BytesIO(z.read('core.pt')),map_location='cpu',weights_only=False)
    return torch.load(path,map_location='cpu',weights_only=False)
