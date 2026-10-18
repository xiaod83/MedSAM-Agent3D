# Checkpoints

The preferred location for SAM weights is `checkpoints/sam_med3d_turbo.pth`.

The existing `sam_med3d_turbo.pth` at the project root could not be moved because the filesystem denied access. It is preserved there, and default inference still falls back to that path. Move it here from your normal file manager when access is available, or pass `--sam-ckpt PATH`.

Weights are excluded from Git. No model weights were deleted or altered during cleanup.
