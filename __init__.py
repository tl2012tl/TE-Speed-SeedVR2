from .nodes import TESpeedSeedVR2, TESpeedSeedVR2Official
from .vae_speed import TESpeedSeedVR2VAE
from .integrated import TESpeedSeedVR2Loader, TESpeedSeedVR2Integrated


# Nodes shown in the ComfyUI menu.
#
# Hidden (still imported and functional, just not registered as menu nodes):
#   - TESpeedSeedVR2         : legacy accelerator for the third-party SeedVR2
#                              node package
#   - TESpeedSeedVR2VAE      : standalone VAE optimizer for the third-party path
#   - TESpeedSeedVR2Official : low-level accelerator, used internally by the
#                              one-node workflow, so it must stay importable
NODE_CLASS_MAPPINGS = {
    "TESpeedSeedVR2Loader": TESpeedSeedVR2Loader,
    "TESpeedSeedVR2Integrated": TESpeedSeedVR2Integrated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedSeedVR2Loader": "TE-Speed-SeedVR2 Loader",
    "TESpeedSeedVR2Integrated": "TE-Speed-SeedVR2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
