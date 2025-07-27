# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.offloading.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.offloading.worker.worker import TransferFunction

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class OffloadingSpec(ABC):
    """Spec for an offloading connector"""

    def __init__(self, vllm_config: "VllmConfig"):
        logger.warning(
            "Initializing OffloadingSpec. This API is experimental and "
            "subject to change in the future as we iterate the design.")
        self._vllm_config = vllm_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config

        self.gpu_block_size = vllm_config.cache_config.block_size
        self.offloaded_block_size = int(
            extra_config.get("block_size", self.gpu_block_size))

    @abstractmethod
    def create_manager(self) -> OffloadingManager:
        """
        Create an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        """
        pass

    @abstractmethod
    def get_transfer_functions(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> list[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                    TransferFunction]]:
        """
        Get transfer functions along with their respective src and dst types.

        Args:
            kv_caches: A dictionary of layer_name -> gpu_kv_cache tensor.

        Returns:
            A list of (src_type, dst_type, transfer_function).
        """
        pass
