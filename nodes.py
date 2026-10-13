import inspect
import logging
import math
import os
import sys
import threading

import torch
import comfy.patcher_extension
from comfy_execution.utils import get_executing_context


_CONFIG_KEY = "te_speed_seedvr2"
_DEBUG_ENV = "TE_SPEED_SEEDVR2_DEBUG"
_INTEGRATION_LOCK = threading.RLock()
_RUN_LOCK = threading.RLock()
_CONFIGS = {}
_OFFICIAL_PATCH_KEY = "te_speed_seedvr2_official"


_PROFILES = {
    "quality": {
        "range": (0.34, 0.66),
        "delta_factor": 1.20,
        "max_consecutive": 1,
        "refresh_interval": 2,
        "rms_drift": 0.08,
        "scale_min": 0.94,
        "scale_max": 1.06,
    },
    "balanced": {
        "range": (0.25, 0.75),
        "delta_factor": 1.60,
        "max_consecutive": 1,
        "refresh_interval": 3,
        "rms_drift": 0.14,
        "scale_min": 0.90,
        "scale_max": 1.10,
    },
    "speed": {
        "range": (0.19, 0.81),
        "delta_factor": 2.20,
        "max_consecutive": 2,
        "refresh_interval": 4,
        "rms_drift": 0.22,
        "scale_min": 0.86,
        "scale_max": 1.14,
    },
}


def _sample_signature(tensor, sample_count=4096):
    flat = tensor.detach().reshape(-1)
    stride = max(1, math.ceil(flat.numel() / sample_count))
    return flat[::stride][:sample_count].float()


def _rms_tensor(tensor):
    sample = _sample_signature(tensor, 2048)
    return torch.sqrt(sample.square().mean() + 1e-12)


def _relative_change_tensor(current, reference):
    denominator = reference.abs().mean().clamp_min(1e-6)
    return (current - reference).abs().mean() / denominator


def _sync_scalars(*values):
    """Transfer several scalar metrics with one device synchronization."""
    if not values:
        return ()
    return tuple(float(value) for value in torch.stack(values).detach().cpu().tolist())


def _relative_scalar(current, reference):
    return abs(current / max(reference, 1e-8) - 1.0)


def _unwrap_dit(model):
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        if hasattr(current, "dit_model"):
            current = current.dit_model
            continue
        break
    return current


class _FrameBlockCache:
    def __init__(self, blocks, threshold, mode):
        self.blocks = list(blocks)
        self.block_count = len(self.blocks)
        self.indices = {id(block): index for index, block in enumerate(self.blocks)}
        self.threshold = float(threshold)
        self.profile = _PROFILES[mode]

        start_ratio, end_ratio = self.profile["range"]
        self.skip_start = max(1, min(self.block_count - 2, round(self.block_count * start_ratio)))
        self.skip_end = max(
            self.skip_start + 1,
            min(self.block_count - 1, round(self.block_count * end_ratio)),
        )

        self.total_forwards = 0
        self.full_forwards = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.reject_counts = {}
        self._clear_cache()
        self._clear_forward()

    def reset(self):
        self.total_forwards = 0
        self.full_forwards = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.reject_counts = {}
        self._clear_cache()
        self._clear_forward()

    def _reject(self, reason):
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def _clear_cache(self):
        self.shape_key = None
        self.signature = None
        self.previous_frame_signature = None
        self.delta_ema = None
        self.vid_residual = None
        self.txt_residual = None
        self.vid_before_rms = 0.0
        self.txt_before_rms = 0.0
        self.vid_after_rms = 0.0
        self.txt_after_rms = 0.0
        self.consecutive_hits = 0
        self.frames_since_full = 0

    def _clear_forward(self):
        self.skip_active = False
        self.full_pending = False
        self.current_signature = None
        self.base_vid = None
        self.base_txt = None
        self.current_vid_before_rms = 0.0
        self.current_txt_before_rms = 0.0

    def begin_forward(self, vid, txt):
        self._clear_forward()
        self.total_forwards += 1
        shape_key = (
            tuple(vid.shape),
            tuple(txt.shape),
            vid.dtype,
            txt.dtype,
            vid.device.type,
            vid.device.index,
        )
        if self.shape_key is not None and shape_key != self.shape_key:
            self._reject("shape_reset")
            self._clear_cache()
        self.shape_key = shape_key

    def _start_full(self, vid, txt, signature, vid_rms=None, txt_rms=None):
        self.full_pending = True
        self.current_signature = signature
        self.base_vid = vid.detach().clone()
        self.base_txt = txt.detach().clone()
        if vid_rms is None or txt_rms is None:
            vid_rms, txt_rms = _sync_scalars(_rms_tensor(vid), _rms_tensor(txt))
        self.current_vid_before_rms = vid_rms
        self.current_txt_before_rms = txt_rms

    def try_cached(self, vid, txt):
        signature = _sample_signature(vid)
        reasons = []

        metric_tensors = [_rms_tensor(vid), _rms_tensor(txt)]
        adjacent_index = None
        if (
            self.previous_frame_signature is not None
            and self.previous_frame_signature.shape == signature.shape
        ):
            adjacent_index = len(metric_tensors)
            metric_tensors.append(
                _relative_change_tensor(signature, self.previous_frame_signature)
            )
        anchor_index = None
        if self.signature is not None and self.signature.shape == signature.shape:
            anchor_index = len(metric_tensors)
            metric_tensors.append(_relative_change_tensor(signature, self.signature))

        metrics = _sync_scalars(*metric_tensors)
        current_vid_rms, current_txt_rms = metrics[:2]
        adjacent_delta = metrics[adjacent_index] if adjacent_index is not None else None
        anchor_delta = metrics[anchor_index] if anchor_index is not None else None
        baseline = self.delta_ema
        self.previous_frame_signature = signature

        if self.signature is None or self.vid_residual is None or self.txt_residual is None:
            reasons.append("missing_residual")
        elif self.signature.shape != signature.shape:
            reasons.append("signature_shape")
        elif baseline is None or adjacent_delta is None:
            reasons.append("delta_calibration")
        else:
            frame_gap = self.consecutive_hits + 1
            delta_limit = max(
                self.threshold,
                baseline * self.profile["delta_factor"] * frame_gap,
            )
            if (
                not math.isfinite(anchor_delta)
                or not math.isfinite(adjacent_delta)
                or anchor_delta > delta_limit
                or adjacent_delta > max(self.threshold, baseline * self.profile["delta_factor"])
            ):
                reasons.append("frame_delta")

        if adjacent_delta is not None and math.isfinite(adjacent_delta):
            if self.delta_ema is None:
                self.delta_ema = adjacent_delta
            else:
                outlier_limit = max(self.threshold * 4.0, self.delta_ema * 3.0)
                if adjacent_delta <= outlier_limit:
                    self.delta_ema = self.delta_ema * 0.80 + adjacent_delta * 0.20

        if self.consecutive_hits >= self.profile["max_consecutive"]:
            reasons.append("mcs")
        if self.frames_since_full >= self.profile["refresh_interval"]:
            reasons.append("refresh")

        if reasons:
            for reason in reasons:
                self._reject(reason)
            self._start_full(
                vid, txt, signature, current_vid_rms, current_txt_rms
            )
            return None

        vid_scale = min(
            self.profile["scale_max"],
            max(self.profile["scale_min"], current_vid_rms / max(self.vid_before_rms, 1e-8)),
        )
        txt_scale = min(
            self.profile["scale_max"],
            max(self.profile["scale_min"], current_txt_rms / max(self.txt_before_rms, 1e-8)),
        )

        vid_residual = self.vid_residual
        txt_residual = self.txt_residual
        if vid_residual.device != vid.device or vid_residual.dtype != vid.dtype:
            vid_residual = vid_residual.to(device=vid.device, dtype=vid.dtype)
        if txt_residual.device != txt.device or txt_residual.dtype != txt.dtype:
            txt_residual = txt_residual.to(device=txt.device, dtype=txt.dtype)

        cached_vid = vid + vid_residual * vid_scale
        cached_txt = txt + txt_residual * txt_scale
        cached_vid_rms, cached_txt_rms = _sync_scalars(
            _rms_tensor(cached_vid), _rms_tensor(cached_txt)
        )
        drift = max(
            _relative_scalar(cached_vid_rms, self.vid_after_rms),
            _relative_scalar(cached_txt_rms, self.txt_after_rms),
        )
        if not math.isfinite(drift) or drift > self.profile["rms_drift"]:
            self._reject("rms_drift")
            self._start_full(
                vid, txt, signature, current_vid_rms, current_txt_rms
            )
            return None

        self.skip_active = True
        self.cache_hits += 1
        self.skipped_blocks += self.skip_end - self.skip_start
        self.consecutive_hits += 1
        self.frames_since_full += 1
        return cached_vid, cached_txt

    def store_full(self, vid, txt):
        if not self.full_pending or self.base_vid is None or self.base_txt is None:
            return
        self.vid_residual = (vid.detach() - self.base_vid).detach()
        self.txt_residual = (txt.detach() - self.base_txt).detach()
        self.signature = self.current_signature.detach()
        self.vid_before_rms = self.current_vid_before_rms
        self.txt_before_rms = self.current_txt_before_rms
        self.vid_after_rms, self.txt_after_rms = _sync_scalars(
            _rms_tensor(vid), _rms_tensor(txt)
        )
        self.full_forwards += 1
        self.consecutive_hits = 0
        self.frames_since_full = 0
        self.full_pending = False
        self.base_vid = None
        self.base_txt = None

    def call_block(self, block, original, vid, txt, vid_shape, txt_shape, emb, cache):
        index = self.indices.get(id(block))
        if index is None:
            return original(block, vid, txt, vid_shape, txt_shape, emb, cache)

        if index == 0:
            self.begin_forward(vid, txt)

        if index == self.skip_start:
            cached = self.try_cached(vid, txt)
            if cached is not None:
                return cached[0], cached[1], vid_shape, txt_shape

        if self.skip_active and self.skip_start < index < self.skip_end:
            if index == self.skip_end - 1:
                self.skip_active = False
            return vid, txt, vid_shape, txt_shape

        result = original(block, vid, txt, vid_shape, txt_shape, emb, cache)
        if index == self.skip_end - 1 and self.full_pending:
            self.store_full(result[0], result[1])
        return result

    def call_official_block(self, index, args, original_block):
        vid = args["vid"]
        txt = args["txt"]

        if index == 0:
            self.begin_forward(vid, txt)

        if index == self.skip_start:
            cached = self.try_cached(vid, txt)
            if cached is not None:
                return {
                    "vid": cached[0],
                    "txt": cached[1],
                    "vid_shape": args["vid_shape"],
                    "txt_shape": args["txt_shape"],
                }

        if self.skip_active and self.skip_start < index < self.skip_end:
            if index == self.skip_end - 1:
                self.skip_active = False
            return {
                "vid": vid,
                "txt": txt,
                "vid_shape": args["vid_shape"],
                "txt_shape": args["txt_shape"],
            }

        result = original_block(args)
        if index == self.skip_end - 1 and self.full_pending:
            self.store_full(result["vid"], result["txt"])
        return result

    def report(self):
        if not self.total_forwards:
            return
        total_blocks = self.total_forwards * self.block_count
        acceleration = 100.0 * self.skipped_blocks / total_blocks
        logging.info("TE-Speed-SeedVR2: acceleration %.1f%%", acceleration)
        if acceleration == 0.0 or os.environ.get(_DEBUG_ENV) == "1":
            details = ", ".join(
                f"{name}={count}" for name, count in sorted(self.reject_counts.items())
            ) or "none"
            logging.info("TE-Speed-SeedVR2: diagnostics %s", details)

    def finish(self):
        self.report()
        self._clear_cache()
        self._clear_forward()


def _compatible_block_class(block_class):
    try:
        params = tuple(inspect.signature(block_class.forward).parameters)
    except (TypeError, ValueError):
        return False
    expected = ("self", "vid", "txt", "vid_shape", "txt_shape", "emb", "cache")
    return params[:len(expected)] == expected


def _patch_block_classes(controller):
    patches = []
    for block_class in {type(block) for block in controller.blocks}:
        if not _compatible_block_class(block_class):
            raise RuntimeError(
                f"TE-Speed-SeedVR2: unsupported block forward version: {block_class.__name__}"
            )
        original = block_class.forward

        def patched(block, vid, txt, vid_shape, txt_shape, emb, cache, _original=original):
            return controller.call_block(
                block, _original, vid, txt, vid_shape, txt_shape, emb, cache
            )

        patched.__name__ = "te_speed_seedvr2_block_forward"
        patched._te_speed_original = original
        block_class.forward = patched
        patches.append((block_class, original, patched))
    return patches


def _restore_block_classes(patches):
    for block_class, original, patched in reversed(patches):
        if block_class.forward is patched:
            block_class.forward = original


class _OfficialBlockPatch:
    def __init__(self, controller, index):
        self.controller = controller
        self.index = index

    def __call__(self, args, extra_options):
        original_block = extra_options.get("original_block")
        if original_block is None:
            raise RuntimeError("TE-Speed-SeedVR2: missing official SeedVR2 block callback")
        return self.controller.call_official_block(self.index, args, original_block)


class _OfficialSamplingScope:
    def __init__(self, controller):
        self.controller = controller
        self.scope_key = None
        self.call_index = 0

    def release(self):
        self.controller.finish()
        self.scope_key = None

    def __call__(self, executor, *args, **kwargs):
        context = get_executing_context()
        if context is None:
            scope_key = None
        else:
            # One sampler invocation is one temporal chunk.  The cache compares
            # whole forwards, and consecutive chunks are different temporal
            # windows (measured signature deltas 0.6-1.3), so a hit would add the
            # previous window's residual to the current one and ghost the whole
            # chunk.  Scoping it to a single call keeps reuse inside one chunk,
            # i.e. across sampling steps, where the content really is the same.
            self.call_index += 1
            scope_key = (context.prompt_id, context.node_id, self.call_index)

        with _RUN_LOCK:
            if scope_key is None or scope_key != self.scope_key:
                self.controller.reset()
                self.scope_key = scope_key
            try:
                return executor(*args, **kwargs)
            finally:
                self.controller.report()


def release_seedvr2_cache(model):
    for scope in model.get_wrappers(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _OFFICIAL_PATCH_KEY
    ):
        scope.release()


def _official_seedvr2_blocks(model):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as exc:
        raise TypeError("TE-Speed-SeedVR2 Official requires a ComfyUI MODEL") from exc

    model_module = type(diffusion_model).__module__.replace("\\", "/").lower()
    if type(diffusion_model).__name__ != "NaDiT" or "comfy.ldm.seedvr.model" not in model_module:
        raise ValueError("TE-Speed-SeedVR2 Official only supports native ComfyUI SeedVR2 models")

    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None or len(blocks) < 4:
        raise RuntimeError("TE-Speed-SeedVR2: unsupported native SeedVR2 block layout")
    return blocks


def _has_official_dit_replacements(model, block_count):
    transformer_options = model.model_options.get("transformer_options", {})
    replacements = transformer_options.get("patches_replace", {}).get("dit", {})
    return any(("block", index) in replacements for index in range(block_count))


def _run_accelerated_phase(original, settings, runner, ctx, *args, **kwargs):
    with _RUN_LOCK:
        model = _unwrap_dit(getattr(runner, "dit", None))
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) < 4:
            logging.warning("TE-Speed-SeedVR2: unsupported DiT model; using original path")
            return original(runner, ctx, *args, **kwargs)

        controller = _FrameBlockCache(
            blocks=blocks,
            threshold=settings["threshold"],
            mode=settings["mode"],
        )
        patches = _patch_block_classes(controller)
        try:
            return original(runner, ctx, *args, **kwargs)
        finally:
            _restore_block_classes(patches)
            controller.finish()


def _find_seedvr2_video_module():
    suffix = "/comfyui-seedvr2_videoupscaler/src/interfaces/video_upscaler.py"
    for module in tuple(sys.modules.values()):
        path = str(getattr(module, "__file__", "")).replace("\\", "/").lower()
        if path.endswith(suffix) and hasattr(module, "upscale_all_batches"):
            return module
    return None


def _ensure_integration():
    with _INTEGRATION_LOCK:
        module = _find_seedvr2_video_module()
        if module is None:
            return False

        current = module.upscale_all_batches
        if getattr(current, "_te_speed_seedvr2_wrapper", False):
            current._te_speed_registry = _CONFIGS
            return True

        original = current

        def wrapped(runner, ctx, *args, **kwargs):
            cache_context = ctx.get("cache_context", {}) if isinstance(ctx, dict) else {}
            config_id = str(cache_context.get("dit_id"))
            settings = wrapped._te_speed_registry.get(config_id)

            def run_dit_phase(inner_runner, inner_ctx, *inner_args, **inner_kwargs):
                if settings is None:
                    return original(
                        inner_runner, inner_ctx, *inner_args, **inner_kwargs
                    )
                return _run_accelerated_phase(
                    original,
                    settings,
                    inner_runner,
                    inner_ctx,
                    *inner_args,
                    **inner_kwargs,
                )

            # The VAE optimizer lives in its own module, but decoding shares
            # SeedVR2's upscale_all_batches entry point with the DiT phase.
            # Dispatch both optimizers here to avoid nested global wrappers.
            try:
                from .vae_speed import run_configured_vae_phase
            except Exception:
                return run_dit_phase(runner, ctx, *args, **kwargs)
            return run_configured_vae_phase(
                run_dit_phase, runner, ctx, *args, **kwargs
            )

        wrapped.__name__ = getattr(original, "__name__", "upscale_all_batches")
        wrapped._te_speed_seedvr2_wrapper = True
        wrapped._te_speed_original = original
        wrapped._te_speed_registry = _CONFIGS
        module.upscale_all_batches = wrapped
        return True


class TESpeedSeedVR2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit": ("SEEDVR2_DIT",),
                "processing_control_value": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "mode": (["quality", "balanced", "speed"], {"default": "balanced"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("SEEDVR2_DIT",)
    RETURN_NAMES = ("dit",)
    FUNCTION = "patch"
    CATEGORY = "TE/optimization"
    DESCRIPTION = "SeedVR2 frame-aware DiT acceleration for image sequences and videos."
    EXPERIMENTAL = True

    def patch(self, dit, processing_control_value, mode, unique_id=None):
        if not isinstance(dit, dict):
            raise TypeError("TE-Speed-SeedVR2 requires a SeedVR2 DiT configuration")
        if int(dit.get("blocks_to_swap", 0)) > 0 or bool(dit.get("swap_io_components", False)):
            raise ValueError("TE-Speed-SeedVR2 does not support SeedVR2 BlockSwap")
        if not _ensure_integration():
            raise RuntimeError(
                "TE-Speed-SeedVR2: ComfyUI-SeedVR2_VideoUpscaler is not loaded or unsupported"
            )

        patched = dict(dit)
        original_id = str(dit.get("node_id", "unknown"))
        config_id = f"te-speed-seedvr2:{unique_id}:{original_id}"
        settings = {
            "threshold": float(processing_control_value),
            "mode": str(mode),
        }
        _CONFIGS[config_id] = settings
        patched["node_id"] = config_id
        patched[_CONFIG_KEY] = dict(settings)
        patched["torch_compile_args"] = None
        return (patched,)


class TESpeedSeedVR2Official:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processing_control_value": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "mode": (["quality", "balanced", "speed"], {"default": "balanced"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "TE/optimization"
    DESCRIPTION = "Native ComfyUI SeedVR2 DiT acceleration for temporal latent chunks."
    EXPERIMENTAL = True

    def patch(self, model, processing_control_value, mode):
        blocks = _official_seedvr2_blocks(model)
        block_count = len(blocks)

        patched = model.clone()
        if _has_official_dit_replacements(patched, block_count):
            raise ValueError(
                "TE-Speed-SeedVR2 Official cannot be combined with another SeedVR2 block replacement"
            )

        controller = _FrameBlockCache(
            blocks=blocks,
            threshold=float(processing_control_value),
            mode=str(mode),
        )
        for index in range(block_count):
            patched.set_model_patch_replace(
                _OfficialBlockPatch(controller, index), "dit", "block", index
            )

        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _OFFICIAL_PATCH_KEY
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _OFFICIAL_PATCH_KEY,
            _OfficialSamplingScope(controller),
        )
        return (patched,)


_ensure_integration()
