# Checkpoints

Downloaded LIBERO and RoboTwin FastWAM weights are stored under
`checkpoints/fastwam_release/`; Wan text encoder, tokenizer, and VAE components use their provider
layouts below `checkpoints/`. All weights are ignored by Git.

The official RoboTwin checkpoint comes from <https://huggingface.co/yuanty/fastwam> at revision
`8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`. The model card does not state a weight license, so
download it for reproduction but do not redistribute it without publisher clarification.

| File | Bytes | SHA-256 |
|---|---:|---|
| `fastwam_release/robotwin_uncond_3cam_384.pt` | 12,041,813,092 | `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` |
| `fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json` | 88,715 | `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` |

Run `scripts/resilient/download_robotwin_assets.py --component model` to download and verify these
files. The complete machine-readable inventory, including shared Apache-2.0 Wan components, is in
`manifests/assets.json`.
