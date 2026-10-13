"""Streaming helpers for the native SeedVR2 causal VAE.

ComfyUI has moved the SeedVR2 VAE API around between releases, so this module
only relies on the parts that stayed stable:

* ``use_slicing`` used to toggle temporal slicing.  Newer builds slice purely
  from the ``slicing_*_min_size`` thresholds and no longer define the flag, so
  it is only touched when the build still has it.
* ``self.device`` is no longer assigned in ``__init__``; newer builds derive it
  from the input tensor inside ``encode``/``decode``.  Slices are therefore
  always handed to the model on the VAE device.
* the decode entry point was renamed from ``decode_`` to ``decode``, and the
  newer ``decode`` applies the scale/shift un-scaling itself.  Calling the
  wrong one would either miss or double-apply that un-scaling.
"""

from contextlib import contextmanager, nullcontext
import logging
import sys
import types

import comfy.model_management
import torch

from comfy.ldm.seedvr.vae import (
    BYTEDANCE_VAE_SCALING_FACTOR,
    BYTEDANCE_VAE_SHIFTING_FACTOR,
)


_DEVICE_CONTEXT = getattr(comfy.model_management, "cuda_device_context", None)
# Head-room kept free when deciding to hold the causal caches on the device.
_CACHE_RESERVE_BYTES = 2 * 1024 ** 3
_DEFAULT_CACHE_BYTES_PER_FRAME_PIXEL = 6200


def _device_context(device):
    return _DEVICE_CONTEXT(device) if _DEVICE_CONTEXT is not None else nullcontext()


def _save_attr(obj, name):
    """Return (had_own_value, value) so an attribute can be restored exactly."""
    data = getattr(obj, "__dict__", None)
    if data is not None and name in data:
        return True, data[name]
    return False, None


def _restore_attr(obj, name, saved):
    had_value, value = saved
    if had_value:
        setattr(obj, name, value)
        return
    try:
        delattr(obj, name)
    except AttributeError:
        pass


def _cpu_encode(native, original, *args, **kwargs):
    # The causal VAE state stays on the compute device; only the produced frame
    # data is moved to CPU between slices.
    return original(*args, **kwargs).to("cpu")


def _cpu_decode(native, original, *args, **kwargs):
    return original(*args, **kwargs).to("cpu")


class StreamingSeedVR2VAE:
    """Run native causal VAE slices while keeping video data on CPU."""

    def __init__(self, vae, chunk_size, tile_size=1024, tile_overlap=128, use_tiling=False):
        self.vae = vae
        self.native = vae.first_stage_model
        self.chunk_size = max(4, int(chunk_size))
        self.tile_size = max(256, int(tile_size))
        self.tile_overlap = max(32, int(tile_overlap))
        self.use_tiling = bool(use_tiling)

    def _load(self, memory_used):
        # VAE is called after the DiT sampling phase.  Ask ComfyUI to load only
        # the VAE patcher; the normal scheduler may otherwise keep/reload the
        # previous MODEL and leave its weights on the GPU during decode.
        try:
            comfy.model_management.free_memory(
                1e30, comfy.model_management.get_torch_device(),
                keep_loaded=[self.vae.patcher],
            )
        except Exception:
            pass
        comfy.model_management.load_models_gpu(
            [self.vae.patcher],
            memory_required=memory_used,
            force_full_load=self.vae.disable_offload,
        )

    @property
    def _decode_is_raw(self):
        """True when the caller must un-scale before decoding.

        Older ComfyUI exposes ``decode_``, which takes the raw latent and
        expects the caller to apply the scale/shift.  Newer ComfyUI renamed the
        entry point to ``decode`` and un-scales internally.
        """
        return getattr(self.native, "decode_", None) is not None

    def _decode_latent(self, latent):
        entry = getattr(self.native, "decode_", None)
        if entry is None:
            entry = self.native.decode
        return entry(latent)

    @staticmethod
    def _ranges(length, tile, overlap):
        stride = max(1, tile - overlap)
        values = []
        for start in range(0, length, stride):
            end = min(start + tile, length)
            if start > 0 and end - start <= overlap:
                continue
            values.append((start, end))
        return values

    @staticmethod
    def _mask(height, width, top, bottom, left, right, overlap, dtype):
        mask_h = torch.ones(height, dtype=dtype)
        mask_w = torch.ones(width, dtype=dtype)
        if overlap:
            ramp = torch.linspace(0, 1, overlap, dtype=dtype)
            if top:
                mask_h[:overlap] = ramp
            if bottom:
                mask_h[-overlap:] = torch.flip(ramp, (0,))
            if left:
                mask_w[:overlap] = ramp
            if right:
                mask_w[-overlap:] = torch.flip(ramp, (0,))
        return mask_h[:, None] * mask_w[None, :]

    def _spatial_tiles(self, value, *, decode):
        _, _, depth, height, width = value.shape
        if decode:
            scale = 8
            tile_h = max(1, self.tile_size // scale)
            tile_w = max(1, self.tile_size // scale)
            overlap = max(0, self.tile_overlap // scale)
            target = (max(1, (depth - 1) * 4 + 1), height * scale, width * scale)
        else:
            scale = 8
            tile_h = self.tile_size
            tile_w = self.tile_size
            overlap = self.tile_overlap
            target = ((depth + 3) // 4, (height + scale - 1) // scale, (width + scale - 1) // scale)

        y_ranges = self._ranges(height, tile_h, overlap)
        x_ranges = self._ranges(width, tile_w, overlap)
        result = None
        weight = None
        with self._cpu_outputs(decode=decode):
            for y_start, y_end in y_ranges:
                for x_start, x_end in x_ranges:
                    tile = value[:, :, :, y_start:y_end, x_start:x_end].contiguous()
                    tile = tile.to(device=self.vae.device, dtype=self.vae.vae_dtype)
                    if decode:
                        output = self._decode_latent(tile)
                    else:
                        output = self.native.encode(tile)
                    if output.ndim == 4:
                        output = output.unsqueeze(2)
                    output = output.to("cpu")
                    if result is None:
                        result = torch.zeros(
                            (output.shape[0], output.shape[1], target[0], target[1], target[2]),
                            dtype=output.dtype,
                        )
                        weight = torch.zeros((1, 1, 1, target[1], target[2]), dtype=output.dtype)
                    if decode:
                        out_y = y_start * scale
                        out_x = x_start * scale
                    else:
                        out_y = y_start // scale
                        out_x = x_start // scale
                    valid_depth = min(output.shape[2], target[0])
                    end_y = min(out_y + output.shape[3], target[1])
                    end_x = min(out_x + output.shape[4], target[2])
                    mask_overlap = overlap * scale if decode else max(0, overlap // scale)
                    mask_overlap = min(mask_overlap, output.shape[3] // 2, output.shape[4] // 2)
                    mask = self._mask(
                        output.shape[3], output.shape[4], y_start > 0, y_end < height,
                        x_start > 0, x_end < width, mask_overlap, output.dtype,
                    )
                    result[:, :, :valid_depth, out_y:end_y, out_x:end_x] += output[:, :, :valid_depth, :end_y - out_y, :end_x - out_x] * mask[:end_y - out_y, :end_x - out_x]
                    weight[:, :, :, out_y:end_y, out_x:end_x] += mask[:end_y - out_y, :end_x - out_x]
                    del tile, output, mask
        return result / weight.clamp_min(1e-6)

    @contextmanager
    def _cpu_outputs(self, *, decode=False):
        native = self.native
        original = native._decode if decode else native._encode
        wrapper = _cpu_decode if decode else _cpu_encode

        def bound(instance, *args, **kwargs):
            return wrapper(instance, original, *args, **kwargs)

        replacement = types.MethodType(bound, native)
        if decode:
            native._decode = replacement
        else:
            native._encode = replacement

        # Only touch attributes this ComfyUI build actually defines.  Newer
        # builds dropped ``use_slicing`` and assign ``device`` from the input
        # tensor, so restoring a stale value would break them.
        names = ("slicing_sample_min_size", "slicing_latent_min_size", "use_slicing", "device")
        saved = {name: _save_attr(native, name) for name in names}
        if hasattr(native, "use_slicing"):
            native.use_slicing = True
        native.slicing_sample_min_size = self.chunk_size
        native.slicing_latent_min_size = max(1, self.chunk_size // 4)
        native.device = self.vae.device
        try:
            yield
        finally:
            if decode:
                native._decode = original
            else:
                native._encode = original
            for name, value in saved.items():
                _restore_attr(native, name, value)

    @contextmanager
    def _cache_offload_scope(self):
        """Keep the causal VAE caches on the device when there is room for them.

        Newer ComfyUI packs the per-convolution temporal tails and holds them in
        pinned host memory whenever the pin budget allows.  That suits a single
        full-frame pass, but this decoder walks one spatial tile at a time, so
        the policy builds a separate offloaded cache per tile and the host round
        trips dominate: the same clip decoded ~65% slower.  Tiling already
        bounds the activations, so when the tile's caches still fit, parking them
        in host memory only costs time.
        """
        if not self.use_tiling:
            yield
            return
        module = sys.modules.get("comfy.ldm.seedvr.vae")
        hook = getattr(module, "_offload_caches_for", None)
        if hook is None:
            # Older ComfyUI has no host offload for these caches.
            yield
            return
        per_pixel = getattr(
            module, "SEEDVR2_CACHE_BYTES_PER_FRAME_PIXEL",
            _DEFAULT_CACHE_BYTES_PER_FRAME_PIXEL,
        )
        cache_bytes = float(self.tile_size) ** 2 * float(per_pixel)
        try:
            free = float(comfy.model_management.get_free_memory(self.vae.device))
        except Exception:
            free = 0.0
        if free - cache_bytes <= _CACHE_RESERVE_BYTES:
            logging.info(
                "TE-Speed-SeedVR2: VAE causal caches stay host-offloaded "
                "(free=%.2f GiB, tile caches=%.2f GiB)",
                free / 1024 ** 3, cache_bytes / 1024 ** 3,
            )
            yield
            return
        module._offload_caches_for = lambda *args, **kwargs: False
        logging.info(
            "TE-Speed-SeedVR2: VAE causal caches kept on device "
            "(free=%.2f GiB, tile caches=%.2f GiB)",
            free / 1024 ** 3, cache_bytes / 1024 ** 3,
        )
        try:
            yield
        finally:
            module._offload_caches_for = hook

    def encode(self, frames):
        pixels = frames.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        pixels = self.vae.process_input(pixels).to(self.vae.vae_dtype)
        shape = (1, 3, min(pixels.shape[2], self.chunk_size + 1), pixels.shape[3], pixels.shape[4])
        if self.use_tiling:
            shape = (*shape[:3], min(shape[3], self.tile_size), min(shape[4], self.tile_size))
        self._load(self.vae.memory_used_encode(shape, self.vae.vae_dtype))
        with _device_context(self.vae.device), self._cache_offload_scope():
            if self.use_tiling:
                encoded = self._spatial_tiles(pixels, decode=False)
            else:
                with self._cpu_outputs():
                    # Hand the input over on the VAE device: newer ComfyUI
                    # derives ``self.device`` from the tensor it receives.
                    encoded = self.native.encode(pixels.to(self.vae.device))
        return self.native.comfy_format_encoded(encoded).to("cpu")

    def decode(self, latent):
        if self._decode_is_raw:
            latent = latent.to("cpu") / BYTEDANCE_VAE_SCALING_FACTOR + BYTEDANCE_VAE_SHIFTING_FACTOR
        shape = (
            latent.shape[0], latent.shape[1],
            min(latent.shape[2], self.chunk_size // 4 + 1),
            latent.shape[3], latent.shape[4],
        )
        if self.use_tiling:
            shape = (*shape[:3], min(shape[3], self.tile_size // 8), min(shape[4], self.tile_size // 8))
        self._load(self.vae.memory_used_decode(shape, self.vae.vae_dtype))
        with _device_context(self.vae.device), self._cache_offload_scope():
            if self.use_tiling:
                decoded = self._spatial_tiles(latent, decode=True)
            else:
                with self._cpu_outputs(decode=True):
                    decoded = self._decode_latent(
                        latent.to(device=self.vae.device, dtype=self.vae.vae_dtype)
                    )
        h, w = decoded.shape[-2:]
        decoded = decoded[..., : h - h % 2, : w - w % 2]
        return self.vae.process_output(decoded.to(self.vae.vae_output_dtype())).movedim(1, -1).to("cpu")


@contextmanager
def streaming_vae(vae, chunk_size, tile_size, overlap, use_tiling):
    yield StreamingSeedVR2VAE(vae, chunk_size, tile_size, overlap, use_tiling)
