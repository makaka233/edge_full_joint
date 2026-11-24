"""Run the DeepSeek state sufficiency and action-value audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .collect_states import main as collect_main
from .evaluate import evaluate
from .offline_predictor import load_groups
from .report import report


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("collect", "evaluate", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.stage in ("collect", "all"):
        import sys

        old = sys.argv
        try:
            sys.argv = [old[0], "--workers", str(args.workers)]
            collect_main()
        finally:
            sys.argv = old
    if args.stage in ("evaluate", "report", "all"):
        if args.stage in ("evaluate", "all"):
            groups = load_groups(RESULTS / "groups")
            payload = evaluate(groups)
            (RESULTS / "evaluation.json").write_text(
                __import__("json").dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf8",
            )
        if args.stage in ("report", "all"):
            report(RESULTS)


if __name__ == "__main__":
    main()
