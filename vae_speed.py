"""Runtime-only SeedVR2 VAE optimizations.

This module deliberately does not use torch.compile and does not reuse VAE
activations between frames.  It changes only execution settings that preserve
the VAE computation itself: causal temporal chunk size and cuDNN algorithm
selection for fixed CUDA shapes.
"""

import logging
import sys
import threading
import types
from typing import Any, Dict, Optional


_INTEGRATION_LOCK = threading.RLock()
_RUN_LOCK = threading.RLock()
_CONFIGS: Dict[str, Dict[str, Any]] = {}
_MODULE = None


_PROFILES = {
    "quality": {"chunk": 4, "min_free_gb": 0.0},
    "balanced": {"chunk": 8, "min_free_gb": 3.0},
    "speed": {"chunk": 12, "min_free_gb": 6.0},
}


def _find_video_module():
    suffix = "/comfyui-seedvr2_videoupscaler/src/interfaces/video_upscaler.py"
    for module in tuple(sys.modules.values()):
        path = str(getattr(module, "__file__", "")).replace("\\", "/").lower()
        if path.endswith(suffix) and hasattr(module, "encode_all_batches"):
            return module
    return None


def _get_memory_device(vae):
    # SeedVR2 stores this on InflatedCausalConv3d modules.  "same" is the
    # normal default and is safe for models that do not expose the attribute.
    try:
        for module in vae.modules():
            value = getattr(module, "memory_device", None)
            if value is not None:
                return value
    except Exception:
        pass
    return "same"


def _current_split(vae):
    value = getattr(vae, "slicing_sample_min_size", None)
    try:
        return int(value) if value is not None else 4
    except (TypeError, ValueError):
        return 4


def _round_chunk(value: int) -> int:
    # SeedVR2's temporal factor is normally four.  Multiples of four preserve
    # the intended encode/decode slicing relationship.
    value = max(4, int(value))
    value = max(4, (value // 4) * 4)
    return value


def _free_vram_gb(device) -> Optional[float]:
    try:
        import torch
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return float(free_bytes) / (1024.0 ** 3)
    except Exception:
        return None


def _choose_chunk(vae, sample, settings):
    profile = _PROFILES[settings["mode"]]
    frame_count = None
    try:
        tensor = sample[0] if isinstance(sample, (list, tuple)) and sample else sample
        if hasattr(tensor, "ndim"):
            # SeedVR2 passes both video and latent samples as [C,T,H,W].
            # A grouped sample is [B,C,T,H,W].
            if tensor.ndim == 5:  # B,C,T,H,W
                frame_count = int(tensor.shape[2])
            elif tensor.ndim == 4:  # C,T,H,W
                frame_count = int(tensor.shape[1])
            device = tensor.device
        else:
            device = next(vae.parameters()).device
    except Exception:
        device = None

    requested = settings.get("chunk_size", 0)
    target = int(requested) if requested else int(profile["chunk"])
    free_gb = _free_vram_gb(device) if device is not None else None
    if settings.get("auto_chunk", True) and free_gb is not None:
        # Conservative thresholds.  A larger chunk lowers Python/slicing
        # overhead but increases peak activation memory.
        if free_gb < 2.0:
            target = 4
        elif settings["mode"] == "balanced" and free_gb < profile["min_free_gb"]:
            target = 4
        elif settings["mode"] == "speed" and free_gb < profile["min_free_gb"]:
            target = min(target, 8)
    return _round_chunk(target), frame_count


def _configure_vae(vae, sample, settings):
    if vae is None or not hasattr(vae, "set_causal_slicing"):
        return None

    previous = (
        _current_split(vae),
        _get_memory_device(vae),
        bool(getattr(vae, "use_slicing", True)),
    )
    chosen, frame_count = _choose_chunk(vae, sample, settings)
    try:
        vae.set_causal_slicing(
            split_size=chosen,
            memory_device=previous[1],
        )
    except Exception as exc:
        logging.debug("TE-Speed-SeedVR2-VAE: keeping original slicing: %s", exc)
        return None
    return previous


def _restore_vae(vae, previous):
    if previous is None or vae is None or not hasattr(vae, "set_causal_slicing"):
        return
    split_size, memory_device, use_slicing = previous
    try:
        vae.set_causal_slicing(
            split_size=split_size if use_slicing else None,
            memory_device=memory_device,
        )
    except Exception:
        pass


def _run_phase(original, settings, runner, ctx, *args, **kwargs):
    with _RUN_LOCK:
        vae = getattr(runner, "vae", None)
        previous = None
        configured = False
        old_benchmark = None
        saved_instance_methods = {}
        try:
            # The VAE calls are wrapped below so the actual tensor shape is
            # available before choosing a safe temporal chunk size.
            if vae is not None:
                import torch
                if torch.cuda.is_available():
                    old_benchmark = torch.backends.cudnn.benchmark
                    torch.backends.cudnn.benchmark = True

                old_encode = getattr(runner, "vae_encode", None)
                old_decode = getattr(runner, "vae_decode", None)
                for name in ("vae_encode", "vae_decode"):
                    if name in getattr(runner, "__dict__", {}):
                        saved_instance_methods[name] = runner.__dict__[name]

                def fast_encode(instance, samples):
                    nonlocal previous, configured
                    if not configured:
                        previous = _configure_vae(instance.vae, samples, settings)
                        configured = previous is not None
                    return old_encode(samples)

                def fast_decode(instance, latents):
                    nonlocal previous, configured
                    if not configured:
                        previous = _configure_vae(instance.vae, latents, settings)
                        configured = previous is not None
                    return old_decode(latents)

                if old_encode is not None:
                    runner.vae_encode = types.MethodType(fast_encode, runner)
                if old_decode is not None:
                    runner.vae_decode = types.MethodType(fast_decode, runner)

            return original(runner, ctx, *args, **kwargs)
        finally:
            if vae is not None:
                _restore_vae(vae, previous)
            if old_benchmark is not None:
                try:
                    import torch
                    torch.backends.cudnn.benchmark = old_benchmark
                except Exception:
                    pass
            # Remove our instance-level overrides, then restore any instance
            # methods that were already present before this phase.
            for name in ("vae_encode", "vae_decode"):
                if name in getattr(runner, "__dict__", {}):
                    try:
                        delattr(runner, name)
                    except Exception:
                        pass
                if name in saved_instance_methods:
                    try:
                        setattr(runner, name, saved_instance_methods[name])
                    except Exception:
                        pass


def run_configured_vae_phase(original, runner, ctx, *args, **kwargs):
    """Run a SeedVR2 phase with VAE settings selected by ctx.vae_id."""
    cache_context = ctx.get("cache_context", {}) if isinstance(ctx, dict) else {}
    config_id = str(cache_context.get("vae_id"))
    settings = _CONFIGS.get(config_id)
    if settings is None:
        return original(runner, ctx, *args, **kwargs)
    return _run_phase(original, settings, runner, ctx, *args, **kwargs)


def _ensure_integration():
    global _MODULE
    with _INTEGRATION_LOCK:
        module = _find_video_module()
        if module is None:
            return False
        _MODULE = module
        # Decoding is dispatched by nodes.py's existing upscale wrapper so the
        # DiT and VAE nodes never wrap the same function twice.  Only encoding
        # needs a dedicated wrapper here.
        for function_name in ("encode_all_batches",):
            current = getattr(module, function_name, None)
            marker = "_te_speed_seedvr2_vae_wrapper"
            if current is None:
                continue
            if getattr(current, marker, False):
                current._te_speed_vae_registry = _CONFIGS
                continue
            original = current

            def wrapped(runner, ctx, *args, _original=original, **kwargs):
                return run_configured_vae_phase(
                    _original, runner, ctx, *args, **kwargs
                )

            wrapped.__name__ = getattr(original, "__name__", function_name)
            setattr(wrapped, marker, True)
            wrapped._te_speed_vae_original = original
            wrapped._te_speed_vae_registry = _CONFIGS
            setattr(module, function_name, wrapped)
        return True


class TESpeedSeedVR2VAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("SEEDVR2_VAE",),
                "mode": (["quality", "balanced", "speed"], {"default": "balanced"}),
                "chunk_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "step": 4,
                     "tooltip": "0=automatic; otherwise temporal VAE chunk size."},
                ),
                "auto_chunk": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("SEEDVR2_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "patch"
    CATEGORY = "TE/optimization"
    DESCRIPTION = (
        "SeedVR2 VAE runtime optimization without torch.compile. "
        "Uses adaptive causal temporal slicing and fixed-shape cuDNN tuning; "
        "does not reuse VAE activations or alter model weights."
    )
    EXPERIMENTAL = True

    def patch(self, vae, mode, chunk_size=0, auto_chunk=True, unique_id=None):
        if not isinstance(vae, dict):
            raise TypeError("TE-Speed-SeedVR2-VAE requires a SeedVR2 VAE configuration")
        # nodes.py owns the shared decode/DiT entry point; ensure it is active
        # even when the user only adds the VAE optimizer node.
        try:
            from . import nodes as shared_integration
            shared_ready = shared_integration._ensure_integration()
        except Exception:
            shared_ready = False
        if not shared_ready or not _ensure_integration():
            raise RuntimeError(
                "TE-Speed-SeedVR2-VAE: ComfyUI-SeedVR2_VideoUpscaler is not loaded or unsupported"
            )

        original_id = str(vae.get("node_id", "unknown"))
        config_id = f"te-speed-seedvr2-vae:{unique_id}:{original_id}"
        _CONFIGS[config_id] = {
            "mode": str(mode),
            "chunk_size": int(chunk_size),
            "auto_chunk": bool(auto_chunk),
        }
        patched = dict(vae)
        patched["node_id"] = config_id
        patched["torch_compile_args"] = None
        patched["te_speed_seedvr2_vae"] = dict(_CONFIGS[config_id])
        return (patched,)


_ensure_integration()


NODE_CLASS_MAPPINGS = {"TESpeedSeedVR2VAE": TESpeedSeedVR2VAE}
NODE_DISPLAY_NAME_MAPPINGS = {"TESpeedSeedVR2VAE": "TE-Speed-SeedVR2-VAE"}
