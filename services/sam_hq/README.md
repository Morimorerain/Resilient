# Temporary HQ-SAM mask service

This audit-only service implements the same `MaskProvider` response contract as the SAM 3.1
service, but it cannot consume language prompts or track entities. It therefore requires one
simulator-oracle bounding box for every camera, entity, and frame. Results are an upper-bound
engineering audit and are not deployable localization evidence or a substitute for the planned
SAM 3.1 experiment.

The frozen model comes from the official `segment-anything-hq==0.3` package. The checkpoint,
license, canonical location, and checksum are documented in `checkpoints/sam_hq/README.md`.

```bash
bash scripts/resilient/create_samhq_environment.sh .venv-samhq
CUDA_VISIBLE_DEVICES=0 .venv-samhq/bin/python services/sam_hq/server.py \
  --queue-dir data/.cache/task_decoupling_rpc_samhq \
  --checkpoint checkpoints/sam_hq/sam_hq_vit_h.pth
```

Stop the service with `Ctrl-C` after the audit. Queue data and masks are ignored by Git.
