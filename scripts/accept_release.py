"""Isolated ZIP acceptance: source hashes, plots, regression tests and four real smokes."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with ZipFile(args.archive) as archive:
        entries = archive.namelist()
        if any(PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
               or '\\' in name or ':' in name for name in entries):
            raise ValueError('Unexpected ZIP path')
        if any(Path(name).suffix.lower() in ('.zip', '.pt', '.pth', '.ckpt') for name in entries):
            raise ValueError('Weight or nested ZIP in research release')
        archive.extractall(root)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf8')

    def run(*command):
        print('> '+' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=root, env=env, check=True)

    run('-m', 'scripts.verify_package')
    run('-m', 'scripts.audit_research_archive')
    run('-m', 'unittest', 'discover', '-s', 'tests', '-v')
    run('-m', 'edge_d.cli', 'plot', '--out', 'runs/replot')
    original = {path.relative_to(root/'evidence/figures'): path
                for path in (root/'evidence/figures').rglob('*.png')}
    generated = {path.relative_to(root/'runs/replot'): path for path in (root/'runs/replot').rglob('*.png')}
    if not original or original.keys() != generated.keys() or any(
            path.read_bytes() != generated[name].read_bytes() for name, path in original.items()):
        raise AssertionError('Archived and regenerated PNGs differ')
    exp005 = root/'evidence/spike_study/EXP005'
    spike_figures = []
    if exp005.is_dir():
        run('-m', 'scripts.spike_results', '--redraw', 'evidence/spike_study/EXP005', '--out', 'runs/replot_exp005')
        spike_figures = list((exp005/'figures').glob('*.png'))
        if not spike_figures or any(path.read_bytes() != (root/'runs/replot_exp005'/path.name).read_bytes()
                                    for path in spike_figures):
            raise AssertionError('EXP005 PNGs differ')
    exp006 = root/'evidence/generalization/EXP006'
    generalization_figures = []
    if exp006.is_dir():
        destination = root/'runs'/'exp006_replot'
        destination.mkdir(parents=True)
        shutil.copytree(exp006/'raw', destination/'raw')
        run('-m', 'scripts.exp006_report', '--out', 'runs/exp006_replot')
        generalization_figures = list((exp006/'figures').glob('*.png'))
        if not generalization_figures or any(path.read_bytes() != (destination/'figures'/path.name).read_bytes()
                                             for path in generalization_figures):
            raise AssertionError('EXP006 PNGs differ')
    exp007 = root/'evidence/macro_credit/EXP007'
    macro_figures = []
    if exp007.is_dir():
        destination = root/'runs'/'exp007_replot'
        destination.mkdir(parents=True)
        shutil.copytree(exp007/'raw', destination/'raw')
        run('-m', 'scripts.exp007_report', '--out', 'runs/exp007_replot')
        macro_figures = list((exp007/'figures').glob('*.png'))
        if not macro_figures or any(path.read_bytes() != (destination/'figures'/path.name).read_bytes()
                                    for path in macro_figures):
            raise AssertionError('EXP007 PNGs differ')
    review = root/'EXP007_AI_RESEARCH_REVIEW.md'
    review_figures = []
    if review.is_file():
        run('-m', 'scripts.research_review', '--experiment', 'EXP007', '--out', review.name, '--verify')
        run('-m', 'scripts.review_exp007_figures', '--review', review.name, '--out', 'runs/review_figures')
        review_figures = list((exp007/'review_figures').glob('*.png'))
        if len(review_figures) != 3 or any(path.read_bytes() != (root/'runs/review_figures'/path.name).read_bytes()
                                         for path in review_figures):
            raise AssertionError('EXP007 review PNGs differ')
    exp008 = root/'evidence/deployment_learning/EXP008'
    actor_figures = []
    if exp008.is_dir():
        destination = root/'runs'/'exp008_replot'
        destination.mkdir(parents=True)
        shutil.copytree(exp008/'raw', destination/'raw')
        run('-m', 'scripts.exp008_report', '--out', 'runs/exp008_replot')
        actor_figures = list((exp008/'figures').glob('*.png'))
        if len(actor_figures) != 8 or any(path.read_bytes() != (destination/'figures'/path.name).read_bytes()
                                          for path in actor_figures):
            raise AssertionError('EXP008 PNGs differ')
        run('-m', 'scripts.research_review', '--experiment', 'EXP008', '--out', 'AI_RESEARCH_REVIEW.md', '--verify')
    run('-m', 'scripts.verify_allocator_modes')
    smokes = {}
    for mode in ('d-only', 's-only', 'joint', 'periodic'):
        periods = 2 if mode == 'periodic' else 1
        gs = str(2*periods)
        run('-m', 'edge_d.cli', 'placement-train', '--mode', mode, '--periods', str(periods),
            '--gs', gs, '--eval-every', gs, '--fixed-scenes', '1', '--holdout-scenes', '1',
            '--out', f'runs/smoke_{mode}')
        out = root/'runs'/f'smoke_{mode}'
        manifest = json.loads((out/'manifest.json').read_text(encoding='utf8'))
        if sorted(path.name for path in out.glob('*.pt')) != ['best.pt', 'last.pt'] or (
                manifest['training_gs'] != int(gs)) or not list((out/'figures').glob('reward_raw_*.png')):
            raise AssertionError(f'Incomplete real training smoke: {mode}')
        smokes[mode] = {key: manifest[key] for key in ('training_gs', 'd_updates', 's_updates')}
    run('-m', 'scripts.verify_package')
    record = {'status': 'passed', 'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              'identical_regenerated_pngs': (len(original)+len(spike_figures)+len(generalization_figures)
                                             +len(macro_figures)+len(review_figures)+len(actor_figures)),
              'training_smokes': smokes}
    output = root.parent/f'{root.name}_validation.json'
    output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
