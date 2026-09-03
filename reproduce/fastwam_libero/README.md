# FastWAM LIBERO reproduction

This recipe evaluates the released unconditional FastWAM checkpoint with the upstream setting `EVALUATION.sigma_shift=5.0`.

1. Complete the root README environment and asset setup.
2. Run `evaluate_minimal.sh` and inspect its result before starting the full benchmark.
3. Run `evaluate_full.sh`. The manager resumes completed tasks from the output directory.
4. Summarize the result with the upstream `experiments/libero/summarize_results.py` utility.
5. Promote only compact metrics and provenance into `reports/baselines/`.

Do not commit raw rollout videos, worker logs, checkpoints, or simulator assets.

## Validated result

The 2026-09-04 reference run completed all 40 tasks and 2,000 episodes with 1,943 successes (97.15%). It used eight RTX 6000 Ada GPUs, seed 42, 10 inference steps, sigma shift 5.0, CFG 1.0, and the exact assets pinned in `manifests/assets.json`. The failure queue was empty, and independent aggregation matched the upstream summary. See `reports/baselines/fastwam-libero-full.json` for suite metrics and provenance.
