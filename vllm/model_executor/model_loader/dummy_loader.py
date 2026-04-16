# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load model on meta device, then materialize with tiny placeholders
        to avoid GPU OOM for large models."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device
            if load_config.device is None
            else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            # Build model structure on meta device — zero memory.
            with torch.device("meta"):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # Replace every parameter/buffer with a scalar (1-element) tensor
            # on GPU. This uses virtually no memory while keeping the model
            # structure intact for KV cache initialization.
            def _make_placeholder(shape, dtype):
                if len(shape) == 0:
                    # Scalar tensor
                    return torch.zeros((), dtype=dtype, device=target_device)
                return torch.zeros(
                    1, dtype=dtype, device=target_device
                ).expand(shape)

            for name, param in list(model.named_parameters()):
                real = _make_placeholder(param.shape, param.dtype)
                with torch.no_grad():
                    new_param = nn.Parameter(real, requires_grad=False)
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                setattr(mod, parts[-1], new_param)

            for name, buf in list(model.named_buffers()):
                real = _make_placeholder(buf.shape, buf.dtype)
                parts = name.split(".")
                mod = model
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                mod.register_buffer(parts[-1], real)

            logger.info("Dummy model loaded (meta → %s), weights are "
                        "1-element expanded tensors.", target_device)

        return model.eval()

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(layer, model_config)

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        materialize_layer(layer, info)

        for tensor in get_layer_tensors(layer).values():
            initialize_single_dummy_weight(tensor)

        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        info.reset()
