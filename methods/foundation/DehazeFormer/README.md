# DehazeFormer Placeholder

This release package intentionally does **not** bundle the third-party
DehazeFormer source tree.

The path is reserved only because `scripts/dehaze_training_images.py` can use
DehazeFormer as an optional preprocessing dependency.

Upstream repository:

- <https://github.com/IDKiro/DehazeFormer>

If you want to enable that optional script, replace this placeholder directory
with a recursive clone of the upstream repository:

```bash
rm -rf methods/foundation/DehazeFormer
git clone --recursive https://github.com/IDKiro/DehazeFormer \
  methods/foundation/DehazeFormer
```

Then follow the upstream README to download the official checkpoints and place
them under `methods/foundation/DehazeFormer/saved_models/`, or pass a checkpoint
path explicitly via `--weights`.
