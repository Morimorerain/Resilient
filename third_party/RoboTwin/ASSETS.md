# RoboTwin evaluation assets

The pinned simulator revision expects external assets under `third_party/RoboTwin/assets/`. They
come from <https://huggingface.co/datasets/TianxingChen/RoboTwin2.0> at revision
`981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`. The repository metadata does not declare an asset
license; use the files for evaluation and do not redistribute them without publisher clarification.

| Archive | Installed directory | Bytes | SHA-256 |
|---|---|---:|---|
| `background_texture.zip` | `assets/background_texture/` | 10,970,687,027 | `54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070` |
| `embodiments.zip` | `assets/embodiments/` | 219,859,313 | `6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6` |
| `objects.zip` | `assets/objects/` | 3,737,778,549 | `6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27` |

Run `scripts/resilient/download_robotwin_assets.py --component simulator`. Archives are retained
under the ignored `downloads/robotwin/` directory, verified before extraction, and never committed.
