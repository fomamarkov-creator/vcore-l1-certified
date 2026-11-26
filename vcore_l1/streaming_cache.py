# Copyright (C) 2026 Efim Sergeevich Markov (ef.87@mail.ru)
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

from typing import Any, Dict, List, Optional, Tuple
import torch
from .inference_layer import VCoreKVCacheQuantizer


class VCoreStreamingCache:
    """
    Бесконечный контекстный KV-кэш V-CORE с фиксированным бюджетом памяти:
    удерживает `sink_tokens` (первые 4 токена диалога) + скользящее окно последних
    `window_size` токенов, сжимая всё на лету сферическими кодами E₈ или INT4.
    """

    is_compileable = False

    def __init__(
        self,
        window_size: int = 2048,
        sink_tokens: int = 4,
        group_size: int = 32,
        mode: str = "spherical",
    ):
        self.window_size = max(16, window_size)
        self.sink_tokens = max(0, sink_tokens)
        self.kv_q = VCoreKVCacheQuantizer(group_size=group_size, mode=mode)
        self.key_cache: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.dtype]] = []
        self.value_cache: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.dtype]] = []
        self.layers: List[Any] = []
        self._seen_tokens = 0
        self._stored_tokens = 0

    def _evict_if_needed(self, k_full: torch.Tensor, v_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = k_full.size(-2)
        max_keep = self.sink_tokens + self.window_size
        if seq_len <= max_keep:
            return k_full, v_full
        if self.sink_tokens > 0:
            k_sink = k_full[..., : self.sink_tokens, :]
            v_sink = v_full[..., : self.sink_tokens, :]
            k_tail = k_full[..., -self.window_size :, :]
            v_tail = v_full[..., -self.window_size :, :]
            return torch.cat([k_sink, k_tail], dim=-2), torch.cat([v_sink, v_tail], dim=-2)
        return k_full[..., -self.window_size :, :], v_full[..., -self.window_size :, :]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.size(-2)

        if len(self.key_cache) <= layer_idx:
            k_kept, v_kept = self._evict_if_needed(key_states, value_states)
            if layer_idx == 0:
                self._stored_tokens = k_kept.size(-2)
            self.key_cache.append((*self.kv_q.compress_kv(k_kept), key_states.dtype))
            self.value_cache.append((*self.kv_q.compress_kv(v_kept), value_states.dtype))
            return k_kept, v_kept

        k_prev = self.kv_q.decompress_kv(*self.key_cache[layer_idx])
        v_prev = self.kv_q.decompress_kv(*self.value_cache[layer_idx])
        k_full = torch.cat([k_prev, key_states], dim=-2)
        v_full = torch.cat([v_prev, value_states], dim=-2)

        k_kept, v_kept = self._evict_if_needed(k_full, v_full)
        if layer_idx == 0:
            self._stored_tokens = k_kept.size(-2)

        self.key_cache[layer_idx] = (*self.kv_q.compress_kv(k_kept), key_states.dtype)
        self.value_cache[layer_idx] = (*self.kv_q.compress_kv(v_kept), value_states.dtype)
        return k_kept, v_kept

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._stored_tokens

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.sink_tokens + self.window_size

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> Tuple[int, int]:
        query_length = cache_position if isinstance(cache_position, int) else int(cache_position.shape)
        kv_length = min(self._stored_tokens + query_length, self.sink_tokens + self.window_size)
        return kv_length, 0
