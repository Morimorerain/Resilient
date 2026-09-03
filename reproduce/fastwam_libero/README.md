# FastWAM LIBERO reproduction

This recipe evaluates the released unconditional FastWAM checkpoint with the upstream setting `EVALUATION.sigma_shift=5.0`.

1. Complete the root README environment and asset setup.
2. Run `evaluate_minimal.sh` and inspect its result before starting the full benchmark.
3. Run `evaluate_full.sh`. The manager resumes completed tasks from the output directory.
4. Summarize the result with the upstream `experiments/libero/summarize_results.py` utility.
5. Promote only compact metrics and provenance into `reports/baselines/`.

Do not commit raw rollout videos, worker logs, checkpoints, or simulator assets.
