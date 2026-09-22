# SAM 3.1 mask service

This process is the only part of Resilient that imports Meta SAM 3.1. It runs in the separate
Python 3.12 environment documented in `environment/README.md` and communicates with the Fast-WAM
Python 3.10 process through validated JSON and NumPy files. The model is never trained.

The pinned source is `facebookresearch/sam3` commit
`2345a4ad109ac29c569da749c91d84f10dc08c40`. The expected checkpoint and license are documented in
`checkpoints/sam3/README.md`.

Example, where physical GPU 4 becomes `cuda:0` inside the service:

```bash
CUDA_VISIBLE_DEVICES=4 .venv-sam3/bin/python services/sam3/server.py \
  --queue-dir data/.cache/task_decoupling_rpc \
  --checkpoint checkpoints/sam3/sam3.1_multiplex.pt
```

Stop the service with `Ctrl-C` after all Fast-WAM clients exit. Requests, responses, decoded JPEG
work files, and content-addressed masks remain in ignored `data/.cache/task_decoupling_rpc/`; failed
temporary work directories are removed by the server. The server refuses a checkpoint with a
different SHA-256, a client with a different source/checkpoint identity, or a request whose mask
threshold differs from the running process.

Do not use `--compile` until the uncompiled path passes the documented Gate-1 audit and a local
throughput profile demonstrates a benefit.
