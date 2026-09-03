# Model weights directory

No model weights are required by the initial scaffold.

Store future weights under `models/<model_name>/`. Actual weights and checkpoints are ignored by Git. Before code depends on a model, document its official download URL, license, exact version/revision, filename, expected directory layout, and SHA-256 here and in the root `README.md`.

Example checksum command:

```bash
sha256sum models/<model_name>/<weight_file>
```
