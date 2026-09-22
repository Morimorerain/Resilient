# Temporary HQ-SAM ViT-H checkpoint

This temporary Gate-3 audit uses the official HQ-SAM ViT-H checkpoint:

- Project: <https://github.com/SysCV/sam-hq>
- Official checkpoint: <https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view>
- License: Apache-2.0 in the official source repository
- File: `sam_hq_vit_h.pth`
- Destination: `checkpoints/sam_hq/sam_hq_vit_h.pth`
- Size: `2570940653` bytes
- SHA-256: `a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35`
- Python package: `segment-anything-hq==0.3`

The weight is ignored by Git. Copy or link the locally supplied file into the canonical path, then
verify it:

```bash
mkdir -p checkpoints/sam_hq
ln -s /path/to/sam_hq_vit_h.pth checkpoints/sam_hq/sam_hq_vit_h.pth
sha256sum checkpoints/sam_hq/sam_hq_vit_h.pth
```

The local path is intentionally a placeholder: never commit a machine-specific absolute path.
