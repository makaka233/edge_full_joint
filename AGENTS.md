# Experiment layout and versioning

- Use the project's existing main program and a single established source checkout for every round. Keep code changes in the existing source modules. Use Git branches and commits to identify versions.
- Put the single-file handoff for each round at `research/<EXPERIMENT_ID>/AI_RESEARCH_REVIEW.md`, for example `research/EXP019/AI_RESEARCH_REVIEW.md`. Keep that round's protocol and reproduction notes in the same research folder.
- Write run results and intermediate files beneath the project's shared `runs/results/<EXPERIMENT_ID>/` directory. This includes logs, checkpoints, smoke evidence, evaluation outputs and figures. Distinguish seeds and reruns by filename within that directory.
- Publish researched code, configuration, scripts, documentation and handoff changes to GitHub with the round identifier. Record the commit SHA with the run and in the handoff. Keep raw run files in the designated results directory and link to the evidence from the handoff.
- Run experiments from the main project root. New rounds use the same source checkout and directory layout.
- Before reorganizing old files, inventory their Git status and find code, test and documentation references. Verify a remote recovery point first, keep required regression inputs in `tests/fixtures/`, and record legacy moves in the existing archive index.
