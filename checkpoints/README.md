# Checkpoints

The preferred location for SAM weights is `checkpoints/sam_med3d_turbo.pth`.

For compatibility, default inference also checks `sam_med3d_turbo.pth` at the project root. You can instead supply an explicit location with `--sam-ckpt PATH`.

Obtain weights using the upstream SAM-Med3D instructions or the notebook's checkpoint setup. Weights are excluded from Git.
