from __future__ import annotations

import logging

from typing import Tuple

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    _DeepEPDispatcherImplLowLatency,
    _DeepEPDispatcherImplBase,
    DeepEPDispatcher,
)
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
    is_tbo_enabled,
)

import torch

logger = logging.getLogger(__name__)


class _FuseEPDispatcherImpl(_DeepEPDispatcherImplLowLatency):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fused_moe(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: torch.Tensor,
        gmm1_permuted_weight_scale: torch.Tensor,
        gmm2_weight: torch.Tensor,
        gmm2_weight_scale: torch.Tensor,
    ):
        buffer = self._get_buffer()

        hidden_states, *args = buffer.fused_deep_moe(
            hidden_states,
            topk_idx,
            topk_weights,
            gmm1_permuted_weight,
            gmm1_permuted_weight_scale,
            gmm2_weight,
            gmm2_weight_scale,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )
        return hidden_states


class NpuFuseEPDispatcher(DeepEPDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__(
            group,
            router_topk,
            permute_fusion,
            num_experts,
            num_local_experts,
            hidden_size,
            params_dtype,
            deepep_mode,
            async_finish,
            return_recv_hook,
        )
        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        self._fuseep_dispatcher = _FuseEPDispatcherImpl(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )

    def fused_moe(self, *args, **kwargs) -> Tuple:
        return self._fused_moe_impl(*args, **kwargs)

    def _fused_moe_impl(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        gmm1_permuted_weight: torch.Tensor,
        gmm1_permuted_weight_scale: torch.Tensor,
        gmm2_weight: torch.Tensor,
        gmm2_weight_scale: torch.Tensor,
    ):
        hidden_states = self._get_impl().fused_moe(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            gmm1_permuted_weight=gmm1_permuted_weight,
            gmm1_permuted_weight_scale=gmm1_permuted_weight_scale,
            gmm2_weight=gmm2_weight,
            gmm2_weight_scale=gmm2_weight_scale,
        )
        return hidden_states

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        return self._fuseep_dispatcher
