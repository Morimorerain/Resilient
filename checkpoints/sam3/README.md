# SAM 3.1 checkpoint

Request access to the official gated repository:

- URL: <https://huggingface.co/facebook/sam3.1>
- License: SAM License in the official `facebookresearch/sam3` repository
- File: `sam3.1_multiplex.pt`
- Destination: `checkpoints/sam3/sam3.1_multiplex.pt`
- Size: `3502755717` bytes
- SHA-256: `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`
- Source code: `facebookresearch/sam3` commit
  `2345a4ad109ac29c569da749c91d84f10dc08c40`

Authenticate without placing a token in this repository, then download the checkpoint:

```bash
export HF_HOME="${PWD}/AILOG/caches/huggingface-sam3"
hf auth login
hf download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir checkpoints/sam3
sha256sum checkpoints/sam3/sam3.1_multiplex.pt
```

`HF_HOME` keeps the token and download cache under ignored `AILOG/` on the project data volume.
The weight is local-only and ignored by Git. Review and accept the SAM License before use or
redistribution.
