# Upstream patch register

FastWAM is pinned in `manifests/upstream.json`. Project-specific behavior should live outside `src/fastwam/` whenever possible.

No FastWAM source patch has been applied.

For every future patch, record:

- affected upstream file and function;
- reason an adapter was insufficient;
- Hydra/CLI switch name and default;
- proof that the switch-off path matches the upstream baseline;
- tests and README sections covering the switch.
