from hashlib import sha256

import numpy as np
import pytest

from scripts.exp056_long import verify_prefix
from scripts.exp056_report import METRICS, best_fractional_cpu, effects


def test_paired_replacement_interaction_has_the_correct_baseline():
    arms = {a:{m:x for m in METRICS} for a,x in zip('ABCD',(10.,11.,12.,16.),strict=True)}
    d = effects(arms)
    assert d['B-A']['reward']==1.
    assert d['D-B']['reward']==5.
    assert d['interaction']['reward']==3.


def test_fractional_load_accounts_for_shared_stage_capacity():
    assert best_fractional_cpu(np.array([[1,0],[1,1]]),[3.,3.],[3.,3.])==1.
    assert best_fractional_cpu(np.array([[1,0],[1,1]]),[6.,1.],[3.,3.])==2.


def test_copied_prefix_preserves_append_only_history(tmp_path):
    p = tmp_path/'train.jsonl'
    prefix = b'original records\n'
    h = sha256(prefix).hexdigest()
    p.write_bytes(prefix+b'new record\n')
    verify_prefix(p,len(prefix),h)
    with pytest.raises(ValueError,match='Truncated'):
        verify_prefix(p,100,h)
    p.write_bytes(b'x'+prefix[1:])
    with pytest.raises(ValueError,match='Changed'):
        verify_prefix(p,len(prefix),h)
