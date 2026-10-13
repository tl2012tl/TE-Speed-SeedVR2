"""Self-contained SeedVR2 workflow nodes.

These nodes use ComfyUI's native SeedVR2 implementation directly.  The custom
node only owns orchestration and the optional DiT block cache, so it does not
depend on ComfyUI-SeedVR2_VideoUpscaler (or any private third-party API).
"""

import logging
import os
import threading
import time
from contextlib import contextmanager, ExitStack

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import folder_paths
import comfy.sd
from safetensors import safe_open

from .nodes import TESpeedSeedVR2Official, release_seedvr2_cache
from .streaming_vae import streaming_vae


_SEEDVR2_MODEL_TYPE = "seedvr2"
_SEEDVR2_FOLDER_NAME = "SEEDVR2"
_ATTENTION_LOCK = threading.RLock()
_VAE_CHUNKS = {"quality": 4, "balanced": 8, "speed": 12}
_NATIVE_DIT_REQUIRED_KEYS = {"positive_conditioning", "negative_conditioning"}
_EXTERNAL_CONDITIONING_ASSETS = {
    "positive_conditioning": ("pos_emb.pt", (58, 5120)),
    "negative_conditioning": ("neg_emb.pt", (64, 5120)),
}
# SeedVR2 DiT layouts ComfyUI's native loader recognises.  Any of these means the
# checkpoint is a SeedVR2 DiT, so the bundled conditioning can stand in when the
# file omits it.  The text conditioning is the text encoder's fixed output, so it
# is identical for 3B and 7B (verified byte-for-byte against both).
_SEEDVR2_ARCH_MARKERS = (
    "blocks.31.mlp.all.proj_in_gate.weight",  # 3B, shared MLP keys
    "blocks.35.mlp.all.proj_in_gate.weight",  # 7B, shared MLP keys
)
_SEEDVR2_7B_SEPARATE_MARKER = "blocks.35.mlp.vid.proj_out.weight"  # 7B, separate keys
_SEEDVR2_7B_VID_DIM = 3072
_INCOMPATIBLE_DIT_WARNED = set()
_EXTERNAL_CONDITIONING_CACHE = None
_EXTERNAL_CONDITIONING_LOCK = threading.Lock()
_WINDOW_BATCH_LOCK = threading.RLock()


def _register_seedvr2_model_path():
    """Match ComfyUI-SeedVR2_VideoUpscaler's shared model location."""
    model_dir = os.path.join(folder_paths.models_dir, _SEEDVR2_FOLDER_NAME)
    folder_paths.add_model_folder_path(_SEEDVR2_MODEL_TYPE, model_dir)


def _seedvr2_files():
    _register_seedvr2_model_path()
    files = folder_paths.get_filename_list(_SEEDVR2_MODEL_TYPE)
    # This independent implementation uses ComfyUI's native SeedVR2 loader.
    # Its supported shared checkpoint format is safetensors; third-party GGUF
    # checkpoints remain available through ComfyUI-SeedVR2_VideoUpscaler.
    return [name for name in files if name.lower().endswith(".safetensors")]


def _is_vae_filename(name):
    return "vae" in os.path.basename(name).lower()


def _conditioning_asset_path(filename):
    candidates = (
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(folder_paths.models_dir, _SEEDVR2_FOLDER_NAME, filename),
    )
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _external_conditioning_available():
    return all(
        _conditioning_asset_path(filename) is not None
        for filename, _ in _EXTERNAL_CONDITIONING_ASSETS.values()
    )


def _load_external_conditioning():
    global _EXTERNAL_CONDITIONING_CACHE
    with _EXTERNAL_CONDITIONING_LOCK:
        if _EXTERNAL_CONDITIONING_CACHE is not None:
            return _EXTERNAL_CONDITIONING_CACHE

        loaded = {}
        for key, (filename, expected_shape) in _EXTERNAL_CONDITIONING_ASSETS.items():
            path = _conditioning_asset_path(filename)
            if path is None:
                raise RuntimeError(
                    f"TE-Speed-SeedVR2: required conditioning asset {filename} was not found"
                )
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
                shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
                raise RuntimeError(
                    f"TE-Speed-SeedVR2: invalid {filename}; expected tensor "
                    f"{expected_shape}, got {shape}"
                )
            loaded[key] = tensor.contiguous()

        _EXTERNAL_CONDITIONING_CACHE = loaded
        return loaded


def _seedvr2_dit_architecture(keys, shapes=None):
    """Whether ComfyUI's native loader recognises this checkpoint as a SeedVR2 DiT."""
    if any(marker in keys for marker in _SEEDVR2_ARCH_MARKERS):
        return True
    if shapes is None:
        return False
    shape = shapes.get(_SEEDVR2_7B_SEPARATE_MARKER)
    return shape is not None and shape[0] == _SEEDVR2_7B_VID_DIM


def _native_dit_compatibility(name):
    """Check whether a checkpoint can be adapted to ComfyUI's native SeedVR2 model."""
    _register_seedvr2_model_path()
    path = folder_paths.get_full_path(_SEEDVR2_MODEL_TYPE, name)
    if path is None:
        return False, "file is not available"
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            shapes = {}
            if _SEEDVR2_7B_SEPARATE_MARKER in keys:
                shapes[_SEEDVR2_7B_SEPARATE_MARKER] = tuple(
                    checkpoint.get_slice(_SEEDVR2_7B_SEPARATE_MARKER).get_shape()
                )
    except Exception as exc:
        return False, f"cannot read safetensors header: {exc}"

    missing = sorted(_NATIVE_DIT_REQUIRED_KEYS - keys)
    if not missing:
        return True, None
    if not _seedvr2_dit_architecture(keys, shapes):
        return False, (
            "no embedded " + "/".join(missing) + " and not a recognised SeedVR2 "
            "DiT (3B or 7B), so the bundled pos_emb.pt/neg_emb.pt cannot stand "
            "in for them"
        )
    if not _external_conditioning_available():
        return False, "missing bundled pos_emb.pt/neg_emb.pt conditioning assets"
    return True, None


def _native_dit_files(files):
    compatible = []
    for name in files:
        if _is_vae_filename(name):
            continue
        is_compatible, reason = _native_dit_compatibility(name)
        if is_compatible:
            compatible.append(name)
        elif name not in _INCOMPATIBLE_DIT_WARNED:
            logging.warning(
                "TE-Speed-SeedVR2: hiding incompatible native DiT %s (%s)",
                name,
                reason,
            )
            _INCOMPATIBLE_DIT_WARNED.add(name)
    return compatible


def _load_seedvr2_diffusion_model(model_path, disable_dynamic=False):
    state_dict, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)
    missing = sorted(_NATIVE_DIT_REQUIRED_KEYS - set(state_dict))
    if missing:
        shapes = {}
        if _SEEDVR2_7B_SEPARATE_MARKER in state_dict:
            shapes[_SEEDVR2_7B_SEPARATE_MARKER] = tuple(
                state_dict[_SEEDVR2_7B_SEPARATE_MARKER].shape
            )
        if not _seedvr2_dit_architecture(set(state_dict), shapes):
            raise RuntimeError(
                "TE-Speed-SeedVR2: checkpoint is missing native conditioning tensors "
                "and is not a recognised SeedVR2 DiT (3B or 7B)"
            )
        conditioning = _load_external_conditioning()
        for key in missing:
            state_dict[key] = conditioning[key]
        logging.info(
            "TE-Speed-SeedVR2: injected bundled conditioning into %s (%s)",
            os.path.basename(model_path),
            ", ".join(missing),
        )

    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        metadata=metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(
            f"TE-Speed-SeedVR2: ComfyUI could not detect SeedVR2 model {model_path}"
        )
    model.cached_patcher_init = (
        _load_seedvr2_diffusion_model,
        (model_path,),
    )
    return model


def _sage_var_attention(
    q,
    k,
    v,
    heads,
    cu_seqlens_q,
    cu_seqlens_k,
    *args,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """SeedVR2 variable-length attention backed by ComfyUI's Sage kernel."""
    from comfy.ldm.modules import attention as attention_module
    from comfy.ldm.seedvr import attention as seedvr_attention

    # Newer ComfyUI hands the attention backend opaque containers that have to
    # be unwrapped exactly once; older builds pass the tensors directly.
    container = getattr(attention_module, "AttentionTensorContainer", None)
    if container is not None and isinstance(q, container):
        q, k, v = q.take(), k.take(), v.take()

    q, k, v, head_dim = seedvr_attention._var_attention_qkv(
        q, k, v, heads, skip_reshape
    )
    try:
        from sageattention import sageattn_varlen
    except (ImportError, AttributeError):
        q_splits = torch.tensor_split(q, cu_seqlens_q[1:-1], dim=0)
        k_splits = torch.tensor_split(k, cu_seqlens_k[1:-1], dim=0)
        v_splits = torch.tensor_split(v, cu_seqlens_k[1:-1], dim=0)
        if len(q_splits) != len(k_splits) or len(q_splits) != len(v_splits):
            raise ValueError("SeedVR2 SageAttention received mismatched sequence lengths")
        output = []
        for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
            q_i = q_i.permute(1, 0, 2).unsqueeze(0)
            k_i = k_i.permute(1, 0, 2).unsqueeze(0)
            v_i = v_i.permute(1, 0, 2).unsqueeze(0)
            out_i = attention_module.attention_sage(
                q_i, k_i, v_i, heads, skip_reshape=True, skip_output_reshape=True,
            )
            output.append(out_i.squeeze(0).permute(1, 0, 2))
        output = torch.cat(output, dim=0)
    else:
        # ComfyUI's native SeedVR2 ``cumulative_lengths`` currently returns
        # Python lists.  SageAttention's varlen kernel needs device tensors,
        # and list slicing cannot be subtracted (``list - list``).
        # Normalize both forms here so this also remains compatible with
        # ComfyUI versions that already return tensors.
        cu_seqlens_q = torch.as_tensor(
            cu_seqlens_q, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = torch.as_tensor(
            cu_seqlens_k, dtype=torch.int32, device=k.device
        )
        lengths_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        lengths_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if lengths_q.numel() == 0 or lengths_k.numel() != lengths_q.numel():
            raise ValueError("SeedVR2 SageAttention received mismatched sequence lengths")
        batch = int(getattr(seedvr_attention, "_te_window_batch", 0) or 0)
        if batch > 0 and lengths_q.numel() > batch:
            # Each cumulative-length interval is one independent shifted
            # spatial window. Process groups of windows so Sage never builds a
            # workspace for the whole frame at once.
            pieces = []
            logging.debug(
                "TE-Speed-SeedVR2: Sage window groups=%d windows=%d batch=%d",
                (lengths_q.numel() + batch - 1) // batch,
                lengths_q.numel(), batch,
            )
            for start in range(0, lengths_q.numel(), batch):
                end = min(start + batch, lengths_q.numel())
                q0, q1 = cu_seqlens_q[start], cu_seqlens_q[end]
                k0, k1 = cu_seqlens_k[start], cu_seqlens_k[end]
                pieces.append(sageattn_varlen(
                    q[q0:q1], k[k0:k1], v[k0:k1],
                    cu_seqlens_q[start:end + 1] - q0,
                    cu_seqlens_k[start:end + 1] - k0,
                    int(lengths_q[start:end].max().item()),
                    int(lengths_k[start:end].max().item()),
                    False, 1.0 / (head_dim ** 0.5),
                ))
            output = torch.cat(pieces, dim=0)
        else:
            output = sageattn_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                int(lengths_q.max().item()), int(lengths_k.max().item()),
                False, 1.0 / (head_dim ** 0.5),
            )
    return seedvr_attention._var_attention_output(
        output, heads, head_dim, skip_output_reshape
    )


@contextmanager
def _seedvr_attention_scope(backend):
    if backend != "sage":
        yield "comfy_auto"
        return

    from comfy.ldm.modules import attention as attention_module
    from comfy.ldm.seedvr import attention as seedvr_attention
    from comfy.ldm.seedvr import model as seedvr_model

    if not attention_module.SAGE_ATTENTION_IS_AVAILABLE:
        raise RuntimeError(
            "TE-Speed-SeedVR2: SageAttention was selected but is not installed"
        )

    with _ATTENTION_LOCK:
        previous = (
            seedvr_attention.optimized_var_attention,
            seedvr_model.optimized_var_attention,
            getattr(seedvr_attention, "_te_window_batch", None),
        )
        seedvr_attention.optimized_var_attention = _sage_var_attention
        seedvr_model.optimized_var_attention = _sage_var_attention
        seedvr_attention._te_window_batch = getattr(seedvr_model, "_te_seedvr_window_batch", 0)
        try:
            yield "sage"
        finally:
            seedvr_attention.optimized_var_attention = previous[0]
            seedvr_model.optimized_var_attention = previous[1]
            if previous[2] is None:
                try:
                    delattr(seedvr_attention, "_te_window_batch")
                except AttributeError:
                    pass
            else:
                seedvr_attention._te_window_batch = previous[2]


@contextmanager
def _seedvr_window_batch_scope(window_batch):
    """Batch SeedVR2 spatial windows during each DiT block.

    Native SeedVR2 already partitions the latent into shifted 3D windows, but
    concatenates every window before calling attention.  Processing a bounded
    number of windows at a time lowers the attention workspace while preserving
    the native partition/reverse order and therefore the model's semantics.
    """
    batch = int(window_batch or 0)
    if batch <= 0:
        yield
        return
    from comfy.ldm.seedvr import model as seedvr_model
    cls = seedvr_model.NaSwinAttention
    original = cls.forward

    def batched_forward(self, vid, txt, vid_shape, txt_shape, cache):
        # The native implementation is kept as the reference path.  The
        # attention callback below splits the already-windowed variable-length
        # sequences; no spatial tile is run independently through the DiT.
        return original(self, vid, txt, vid_shape, txt_shape, cache)

    # Window batching is implemented in the varlen attention callback, where
    # all windows are represented as independent cumulative-length sequences.
    # Install the limit in a context-local module attribute to avoid changing
    # ComfyUI's global behavior after the node returns.
    previous = getattr(seedvr_model, "_te_seedvr_window_batch", None)
    seedvr_model._te_seedvr_window_batch = batch
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(seedvr_model, "_te_seedvr_window_batch")
            except AttributeError:
                pass
        else:
            seedvr_model._te_seedvr_window_batch = previous


def _adaptive_vae_chunk(vae, prepared, mode):
    target = _VAE_CHUNKS[mode]
    _, frames, height, width, _ = prepared.shape
    target = min(target, max(4, ((frames - 1) // 4) * 4))

    try:
        free_bytes = comfy.model_management.get_free_memory(vae.device)
        dtype_bytes = max(2, comfy.model_management.dtype_size(vae.vae_dtype))
        reserve = 2 * 1024 ** 3
        budget = max(0, free_bytes - reserve) * 0.55
        while target > 4:
            estimated = max(target, 5) * height * width * 64 * dtype_bytes
            if estimated <= budget:
                break
            target -= 4
    except Exception:
        pass
    # The causal decoder's memory_limit_conv concatenates temporal pieces
    # internally.  Its peak is much higher than the simple activation estimate,
    # especially on 16-GiB cards, so keep the safe upper bound conservative.
    return 4


def _seedvr2_model(model):
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as exc:
        raise TypeError("TE-Speed-SeedVR2 expects a ComfyUI MODEL") from exc
    if type(diffusion_model).__name__ != "NaDiT":
        raise ValueError(
            "TE-Speed-SeedVR2 expects a native ComfyUI SeedVR2 model "
            f"(got {type(diffusion_model).__name__})"
        )
    if not hasattr(diffusion_model, "positive_conditioning"):
        raise ValueError("TE-Speed-SeedVR2 model is missing native conditioning tensors")
    return diffusion_model


def _resize_images(images, max_long_edge):
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(
            "TE-Speed-SeedVR2 expects ComfyUI IMAGE shaped (frames,H,W,C), "
            f"got {tuple(images.shape)}"
        )
    images = images[..., :3].clamp(0.0, 1.0).unsqueeze(0)
    max_long_edge = int(max_long_edge)
    if max_long_edge < 16:
        raise ValueError("TE-Speed-SeedVR2 max_long_edge must be at least 16")
    _, _, h, w, _ = images.shape
    resize_ratio = max_long_edge / max(h, w)
    new_h = max(16, int(round(h * resize_ratio)))
    new_w = max(16, int(round(w * resize_ratio)))
    if (new_h, new_w) != (h, w):
        b, t, h, w, c = images.shape
        flat = images.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(new_h, new_w), mode="bicubic", align_corners=False)
        images = flat.reshape(b, t, c, new_h, new_w).permute(0, 1, 3, 4, 2)

    return images


def _prepare_images(images, max_long_edge):
    images = _resize_images(images, max_long_edge)
    # Native SeedVR2 requires spatial multiples of 16 and pixel-frame count 4n+1.
    b, t, h, w, _ = images.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    if ph or pw:
        flat = images.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w)
        flat = F.pad(flat, (0, pw, 0, ph))
        images = flat.reshape(b, t, 3, h + ph, w + pw).permute(0, 1, 3, 4, 2)
    # A still image already satisfies SeedVR2's 4n+1 temporal layout. Repeating
    # it to five frames makes image restoration roughly five times slower.
    if 1 < t < 5:
        images = torch.cat((images, images[:, -1:].repeat(1, 5 - t, 1, 1, 1)), dim=1)
    elif (t - 1) % 4:
        images = torch.cat((images, images[:, -1:].repeat(1, 4 - ((t - 1) % 4), 1, 1, 1)), dim=1)
    return images


# Calibrated single-decode-tile peak VRAM for the native SeedVR2 3B VAE (fp16).
# ComfyUI's own ``memory_used_decode`` heuristic (~160 bytes per output pixel)
# underestimates the real decoder peak by roughly an order of magnitude, so it
# cannot be trusted to size a spatial tile.  Measured decode peaks on a 16-GiB
# card (1920x1440 input): 512px tile -> 6.4 GiB, 768px -> 11.6 GiB,
# 1024px -> 18.3 GiB.  Fit: peak ~= fixed + per-mpx * (tile/1000)^2 GiB.
_VAE_DECODE_FIXED_GIB = 2.4
_VAE_DECODE_GIB_PER_MPX = 15.2
# Largest first, so the search below returns the biggest tile that still fits.
_VAE_TILE_CHOICES = (1024, 896, 768, 640, 512, 448, 384, 320, 256)
# Scale applied to ComfyUI's optimistic decode estimate when deciding whether
# spatial tiling is required at all.
_VAE_DECODE_ESTIMATE_SCALE = 9.0


def _vae_tile_decode_peak_gib(tile):
    return _VAE_DECODE_FIXED_GIB + _VAE_DECODE_GIB_PER_MPX * (tile / 1000.0) ** 2


def _auto_vae_tile(vae, prepared, requested_tile, requested_overlap):
    """Pick the largest VAE tile whose decode fits the free-VRAM budget.

    Auto mode previously *enlarged* the tile to at least 1024 px, which makes a
    single decode tile peak around 18 GiB and overflows 16-GiB cards; Windows
    then pages to system RAM and the run looks like it hangs.  Instead pick a
    tile that actually fits, and derive the overlap from it.
    """
    try:
        free_bytes = comfy.model_management.get_free_memory(vae.device)
    except Exception:
        free_bytes = 0
    reserve_bytes = 2 * 1024 ** 3
    budget_gib = max(0.0, free_bytes - reserve_bytes) * 0.6 / 1024 ** 3

    tile = _VAE_TILE_CHOICES[-1]
    for candidate in _VAE_TILE_CHOICES:
        if _vae_tile_decode_peak_gib(candidate) <= budget_gib:
            tile = candidate
            break
    # Honour a smaller explicit request as an upper bound.
    tile = max(256, min(tile, int(requested_tile)))
    overlap = min(max(32, int(requested_overlap)), max(32, tile // 8))
    overlap = max(32, min(overlap, tile - 64))
    logging.info(
        "TE-Speed-SeedVR2: VAE auto tile=%d overlap=%d "
        "(budget=%.2f GiB, estimated decode peak=%.2f GiB)",
        tile,
        overlap,
        budget_gib,
        _vae_tile_decode_peak_gib(tile),
    )
    return tile, overlap


def _use_vae_tiling(vae, prepared, setting, tile_size, chunk_size):
    if setting == "enabled":
        return True
    if setting == "disabled":
        return False

    b, t, h, w, c = prepared.shape
    t = min(t, chunk_size + 1)
    try:
        latent_t = max(1, (t - 1) // 4 + 1)
        latent_h = (h + 7) // 8
        latent_w = (w + 7) // 8
        encode_bytes = vae.memory_used_encode((b, c, t, h, w), vae.vae_dtype)
        decode_bytes = vae.memory_used_decode(
            (b, 16, latent_t, latent_h, latent_w), vae.vae_dtype
        )
        # ComfyUI's decode estimate is far too optimistic for this decoder.
        decode_bytes *= _VAE_DECODE_ESTIMATE_SCALE
        estimated_bytes = max(encode_bytes, decode_bytes)
        free_bytes = comfy.model_management.get_free_memory(vae.device)
        reserve_bytes = 2 * 1024 ** 3
        budget_bytes = max(0, free_bytes - reserve_bytes) * 0.7
        use_tiling = max(h, w) > tile_size or estimated_bytes > budget_bytes
        logging.info(
            "TE-Speed-SeedVR2: VAE auto tiling=%s (size=%dx%d, tile=%d, "
            "estimated=%.2f GiB, budget=%.2f GiB)",
            use_tiling,
            w,
            h,
            tile_size,
            estimated_bytes / 1024 ** 3,
            budget_bytes / 1024 ** 3,
        )
        return use_tiling
    except Exception as exc:
        use_tiling = max(h, w) > 2048
        logging.warning(
            "TE-Speed-SeedVR2: VAE memory estimate failed (%s); auto tiling=%s",
            exc,
            use_tiling,
        )
        return use_tiling


def _sampling_chunk_size(model, samples, requested, vram_budget_gb, use_cache):
    device = comfy.model_management.get_torch_device()
    total = comfy.model_management.get_total_memory(device)
    free = comfy.model_management.get_free_memory(device)
    limit = float(vram_budget_gb) * 1024 ** 3 if vram_budget_gb > 0 else total * 0.8
    unloaded_weights = max(0, model.model_size() - model.loaded_size())
    available = max(0, min(free, limit - (total - free)) - unloaded_weights - 1.5 * 1024 ** 3)
    b, _, t, h, w = samples.shape
    megapixels = b * h * w * 64 / 1e6
    bytes_per_frame = 0.55 * megapixels * 1024 ** 3
    cache_overhead = 0.0
    if use_cache:
        dit = _seedvr2_model(model)
        hidden_bytes = b * ((h + 1) // 2) * ((w + 1) // 2) * dit.vid_in.proj.out_features * 2
        # The frame cache keeps a single residual pair (one block's hidden
        # state), not a per-frame cost.  The old "4 * hidden_bytes per frame"
        # reservation was far too conservative: it shrank the chunk and thus
        # added temporal chunk boundaries, which is what causes boundary
        # ghosting.  Reserve a flat, correctly sized overhead instead.
        cache_overhead = 8 * hidden_bytes
    if requested:
        requested = max(1, int(requested))
        frames = 1 if requested == 1 else 4 * ((requested + 2) // 4) + 1
    else:
        # Automatic mode aims for the largest chunk that fits the estimated
        # activation budget, up to a generous cap.  This keeps long videos from
        # being forced into many tiny 5-frame sampler calls.
        frames = min(4 * max(0, t - 1) + 1, 129)
    latent_limit = (frames - 1) // 4 + 1
    usable = max(0.0, available - cache_overhead)
    latent_frames = max(1, min(t, latent_limit, int(usable / max(1, bytes_per_frame))))
    frames = 4 * (latent_frames - 1) + 1
    logging.info(
        "TE-Speed-SeedVR2: DiT chunk=%d frames, VRAM budget=%.2f GiB, "
        "activation budget=%.2f GiB (weights and TE cache included)",
        frames, limit / 1024 ** 3, available / 1024 ** 3,
    )
    return frames


def _effective_window_batch(requested, latent, mode):
    """Avoid excessive Sage/Triton launches on long temporal inputs.

    Window batching saves workspace, but every batch is a separate kernel and
    synchronization point.  Keep the user-selected value for short clips and
    increase it for long clips so runtime does not grow superlinearly.
    """
    requested = int(requested or 0)
    if requested <= 0:
        return 0
    frames = 4 * max(0, int(latent.shape[2]) - 1) + 1
    if frames >= 17:
        return max(requested, 16 if mode == "speed" else 12)
    if frames >= 9:
        return max(requested, 8)
    return requested


def _conditioning(model, latent):
    diffusion_model = _seedvr2_model(model)
    samples = latent["samples"]
    if samples.ndim != 5 or samples.shape[1] != 16:
        raise ValueError(f"SeedVR2 latent must be (B,16,T,H,W), got {tuple(samples.shape)}")
    condition = torch.cat((samples.movedim(1, -1), samples.new_ones(samples.shape[0], samples.shape[2], samples.shape[3], samples.shape[4], 1)), dim=-1)
    condition = condition.movedim(-1, 1).contiguous()
    positive = [[diffusion_model.positive_conditioning.unsqueeze(0), {"condition": condition}]]
    negative = [[diffusion_model.negative_conditioning.unsqueeze(0), {"condition": condition}]]
    return positive, negative



def _non_finite_report(label, tensor):
    """Log the first stage that produced NaN/Inf so a black frame is diagnosable.

    A black output is otherwise indistinguishable from a legitimately dark one;
    naming the stage that went non-finite is what makes it actionable.
    """
    if tensor is None:
        return False
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return False
    logging.warning(
        "TE-Speed-SeedVR2: %s has %d non-finite values out of %d",
        label, int((~finite).sum()), tensor.numel(),
    )
    return True


class TESpeedSeedVR2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        files = _seedvr2_files()
        diffusion = _native_dit_files(files)
        vaes = [name for name in files if _is_vae_filename(name)]
        if not diffusion:
            diffusion = ["<no SeedVR2 DiT found in models/SEEDVR2>"]
        if not vaes:
            vaes = ["<no SeedVR2 VAE found in models/SEEDVR2>"]
        return {"required": {
            "dit_name": (diffusion, ),
            "vae_name": (vaes, ),
        }}

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load"
    CATEGORY = "TE-Speed/SeedVR2"

    def load(self, dit_name, vae_name):
        if dit_name.startswith("<") or vae_name.startswith("<"):
            raise RuntimeError("TE-Speed-SeedVR2: no models found in ComfyUI/models/SEEDVR2")
        compatible, reason = _native_dit_compatibility(dit_name)
        if not compatible:
            raise RuntimeError(
                f"TE-Speed-SeedVR2: {dit_name} cannot be loaded by ComfyUI's native "
                f"SeedVR2 implementation ({reason}). Select a supported 3B "
                "safetensors checkpoint."
            )
        model_path = folder_paths.get_full_path_or_raise(_SEEDVR2_MODEL_TYPE, dit_name)
        vae_path = folder_paths.get_full_path_or_raise(_SEEDVR2_MODEL_TYPE, vae_name)
        model = _load_seedvr2_diffusion_model(model_path)
        vae_state, vae_metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=vae_state, metadata=vae_metadata)
        vae.throw_exception_if_invalid()
        _seedvr2_model(model)
        logging.info("TE-Speed-SeedVR2: loaded %s + %s from models/SEEDVR2", dit_name, vae_name)
        return (model, vae)


class TESpeedSeedVR2Integrated:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "images": ("IMAGE",),
            "max_long_edge": ("INT", {"default": 1920, "min": 16, "max": 16384, "step": 16}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 1, "min": 1, "max": 20}),
            "processing_control_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.005,
                "tooltip": "TE frame-cache threshold; 0 = off (default). The cache compares whole DiT forwards, so it is only safe within one temporal chunk: keep it 0 unless steps > 1."}),
            "mode": (["quality", "balanced", "speed"], {"default": "balanced"}),
            "color_correction_method": (["lab", "wavelet", "adain", "none"], {"default": "lab"}),
            "attention_backend": (["sage", "comfy_auto"], {"default": "sage"}),
        }, "optional": {
            "vae_tiling": (["auto", "enabled", "disabled"], {"default": "auto"}),
            "vae_tile_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64,
                "tooltip": "Spatial VAE tile edge in pixels. 512 is safe on 16-GiB cards; larger tiles need more VRAM. With vae_tiling=auto this is only an upper bound."}),
            "vae_tile_overlap": ("INT", {"default": 64, "min": 32, "max": 256, "step": 32}),
            "video_batch_size": ("INT", {"default": 0, "min": 0, "max": 16385,
                "tooltip": "0=automatic (up to 21 frames). Otherwise a frame limit: 1, 5, 9, 13, ...; reduced if needed to fit VRAM."}),
            "temporal_overlap": ("INT", {"default": 1, "min": 0, "max": 16,
                "tooltip": "Latent frames shared by adjacent DiT chunks; 1 is about 4 video frames. Crossfaded before continuous VAE decoding."}),
            "vram_budget_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                "tooltip": "DiT working-memory budget in GiB, including resident weights/cache. 0 uses 80% of physical VRAM. An estimate, not a hard allocation limit."}),
            "spatial_window_batch": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                "tooltip": "Number of SeedVR2 spatial windows processed per attention call. 0=automatic; lower values reduce VRAM."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "restore"
    CATEGORY = "TE-Speed/SeedVR2"
    DESCRIPTION = "One-node native SeedVR2 restoration with TE frame-cache acceleration."

    def restore(
        self,
        model,
        vae,
        images,
        max_long_edge=1920,
        seed=0,
        steps=1,
        processing_control_value=0.10,
        mode="balanced",
        color_correction_method="lab",
        attention_backend="sage",
        vae_tiling="auto",
        vae_tile_size=1024,
        vae_tile_overlap=128,
        scale=None,
        video_batch_size=0,
        temporal_overlap=1,
        vram_budget_gb=0.0,
        spatial_window_batch=0,
    ):
        # Keep old saved workflows loadable while removing scale from the UI.
        # New workflows always use max_long_edge directly.
        if scale is not None and max_long_edge == 1920:
            _, input_height, input_width, _ = images.shape
            max_long_edge = max(16, int(round(max(input_height, input_width) * float(scale))))
        _seedvr2_model(model)
        if getattr(vae, "latent_channels", None) != 16 or getattr(vae, "latent_dim", None) != 3:
            raise ValueError("TE-Speed-SeedVR2 expects a native SeedVR2 VAE (16 channels, video latent)")
        # Import the optional API lazily so the plugin can still register on
        # installations that do not have ComfyUI's video extension dependencies.
        from comfy_extras.nodes_seedvr import SeedVR2TemporalChunk, SeedVR2TemporalMerge, SeedVR2PostProcessing
        started_at = time.perf_counter()
        original = images.to("cpu")
        prepared = _prepare_images(original, int(max_long_edge))
        prepare_finished_at = time.perf_counter()
        tile_size = max(256, int(vae_tile_size) // 64 * 64)
        tile_overlap = min(max(32, int(vae_tile_overlap)), tile_size - 64)
        vae_chunk = _adaptive_vae_chunk(vae, prepared, mode)
        if vae_tiling == "auto":
            # Pick a tile that actually fits VRAM instead of forcing >=1024 px.
            # A 1024-px decode tile peaks around 18 GiB and overflows 16-GiB
            # cards, where Windows then pages to RAM and the run appears to
            # hang.  Full-frame decode is memory-bounded by the conv slicer but
            # very slow, so only keep it when the whole frame fits one tile.
            tile_size, tile_overlap = _auto_vae_tile(
                vae, prepared, tile_size, tile_overlap
            )
            use_vae_tiling = max(prepared.shape[2], prepared.shape[3]) > tile_size
        else:
            use_vae_tiling = _use_vae_tiling(
                vae, prepared, vae_tiling, tile_size, vae_chunk
            )
        with ExitStack() as stack:
            # One 3x3 SeedVR2 latent window is the smallest useful unit.  Keep
            # the default conservative on 16-GiB cards; users can raise it for
            # speed after confirming VRAM headroom.
            # Disabled by default until the native Sage/Triton build is proven
            # stable on the current Windows/PyTorch combination.  The option
            # remains available for explicit experiments.
            selected_backend = stack.enter_context(_seedvr_attention_scope(attention_backend))
            video_vae = stack.enter_context(streaming_vae(vae, vae_chunk, tile_size, tile_overlap, use_vae_tiling))
            logging.info("TE-Speed-SeedVR2: attention backend=%s", selected_backend)
            logging.info("TE-Speed-SeedVR2: streaming VAE chunk=%d, output storage=CPU", vae_chunk)
            encoded = video_vae.encode(prepared[0])
            _non_finite_report("VAE encode output", encoded)
            encode_finished_at = time.perf_counter()
            requested_window_batch = int(spatial_window_batch or 0)
            effective_window_batch = _effective_window_batch(
                requested_window_batch, encoded, mode
            )
            if effective_window_batch > 0:
                # The attention scope is already active; set the batch limit on
                # the same module object used by _sage_var_attention.
                from comfy.ldm.seedvr import attention as seedvr_attention
                seedvr_attention._te_window_batch = effective_window_batch
            logging.info(
                "TE-Speed-SeedVR2: spatial window batch=%d (requested=%d)",
                effective_window_batch, requested_window_batch,
            )
            latent = {"samples": encoded}
            # The frame cache needs at least two full forwards per chunk to warm
            # up, so with steps=1 it never activates (measured 0% acceleration)
            # and must not reserve chunk memory.  Only pay for it when it can
            # actually help.
            cache_reserve = processing_control_value > 0 and int(steps) > 1
            frames_per_chunk = _sampling_chunk_size(
                model, encoded, video_batch_size, vram_budget_gb, cache_reserve,
            )
            chunks, overlap = SeedVR2TemporalChunk.execute(
                latent, int(temporal_overlap), {"chunking_mode": "manual", "frames_per_chunk": frames_per_chunk}
            ).args
            if len(chunks) > 1:
                logging.warning(
                    "TE-Speed-SeedVR2: video split into %d temporal chunks of %d "
                    "frames.  SeedVR2's temporal windows/rope are relative to the "
                    "sequence it is given, so chunk boundaries can leave a "
                    "ghost/trail on moving content.  Raise video_batch_size (or "
                    "lower max_long_edge) to use fewer chunks.",
                    len(chunks),
                    frames_per_chunk,
                )
            accelerated_model = model
            if len(chunks) > 1 and processing_control_value > 0:
                accelerated_model = TESpeedSeedVR2Official().patch(
                    model, float(processing_control_value), str(mode)
                )[0]
            sampled_chunks = []
            try:
                for chunk_index, chunk in enumerate(chunks):
                    positive, negative = _conditioning(model, chunk)
                    latent_image = chunk["samples"]
                    chunk_seed = (int(seed) + chunk_index) % (2 ** 64)
                    noise = comfy.sample.prepare_noise(latent_image, chunk_seed)
                    sampled = comfy.sample.sample(
                        accelerated_model, noise, int(steps), 1.0, "euler", "simple",
                        positive, negative, latent_image, denoise=1.0, seed=chunk_seed,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    )
                    sampled_chunk = chunk.copy()
                    sampled_chunk["samples"] = sampled.to("cpu")
                    if _non_finite_report(f"DiT sample of chunk {chunk_index}", sampled_chunk["samples"]):
                        logging.warning(
                            "TE-Speed-SeedVR2: the DiT returned non-finite values. On "
                            "this ComfyUI build that has been observed with the --fast "
                            "startup arguments, which route the weights through "
                            "ModelPatcherDynamic (staged, never a full load) instead of "
                            "the classic loader. Remove --fast (and its launcher toggle) "
                            "and retry; the same checkpoint loads correctly without it."
                        )
                    sampled_chunks.append(sampled_chunk)
                    del noise, sampled
            finally:
                if accelerated_model is not model:
                    release_seedvr2_cache(accelerated_model)
                    del accelerated_model
                comfy.model_management.soft_empty_cache()
            sample_finished_at = time.perf_counter()
            sampled = SeedVR2TemporalMerge.execute(
                sampled_chunks, [overlap]
            ).args[0]["samples"]
            _non_finite_report("merged latent", sampled)
            # The DiT is no longer needed during VAE decode.  Keep the phase
            # boundary explicit: ComfyUI otherwise keeps the MODEL resident on
            # the GPU while the VAE allocates its decode activations.  On a
            # 16-GiB card that extra resident model is enough to turn a tiled
            # decode into an OOM even though every video tensor is on CPU.
            # The next sampling call (if there are more temporal chunks) will
            # load the model again through ComfyUI's normal MODEL machinery.
            try:
                comfy.model_management.unload_model_and_clones(model)
                comfy.model_management.soft_empty_cache()
                logging.info(
                    "TE-Speed-SeedVR2: offloaded DiT before VAE decode to "
                    "reserve VRAM for decode activations"
                )
            except Exception as exc:
                logging.warning(
                    "TE-Speed-SeedVR2: could not offload DiT before VAE decode: %s",
                    exc,
                )
            decoded = video_vae.decode(sampled)
            _non_finite_report("VAE decode output", decoded)
            decode_finished_at = time.perf_counter()
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        reference = _resize_images(original, int(max_long_edge))[0]
        result = None
        post_chunk = max(1, min(4, len(original)))
        for start in range(0, len(original), post_chunk):
            end = min(start + post_chunk, len(original))
            corrected = SeedVR2PostProcessing.execute(
                decoded[start:end], reference[start:end], color_correction_method,
            ).args[0].to("cpu")
            _non_finite_report(f"color correction output ({color_correction_method})", corrected)
            if result is None:
                result = corrected.new_empty((len(original), *corrected.shape[1:]))
            result[start:end].copy_(corrected)
        finished_at = time.perf_counter()
        logging.info(
            "TE-Speed-SeedVR2 timings: prepare=%.2fs encode=%.2fs sample=%.2fs "
            "decode=%.2fs post=%.2fs total=%.2fs; image=%s latent=%s",
            prepare_finished_at - started_at,
            encode_finished_at - prepare_finished_at,
            sample_finished_at - encode_finished_at,
            decode_finished_at - sample_finished_at,
            finished_at - decode_finished_at,
            finished_at - started_at,
            tuple(prepared.shape),
            tuple(encoded.shape),
        )
        return (result.clamp_(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "TESpeedSeedVR2Loader": TESpeedSeedVR2Loader,
    "TESpeedSeedVR2Integrated": TESpeedSeedVR2Integrated,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TESpeedSeedVR2Loader": "TE-Speed-SeedVR2 Loader",
    "TESpeedSeedVR2Integrated": "TE-Speed-SeedVR2",
}
