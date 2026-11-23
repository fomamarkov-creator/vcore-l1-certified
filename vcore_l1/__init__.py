from .cuda_quantizer import VCoreTernaryQuantizer, verify_with_lean4_engine
from .matrix_generator import generate_markov_q
from .inference_layer import VCoreLinear, inject_vcore_layers
from .safetensors_quantizer import (
    pack_ternary_to_uint8,
    unpack_uint8_to_ternary,
    repack_safetensors_layerwise,
    repack_sharded_model,
    repack_hf_hub_model,
)

__version__ = "0.3.0"
__all__ = [
    "VCoreTernaryQuantizer",
    "verify_with_lean4_engine",
    "generate_markov_q",
    "VCoreLinear",
    "inject_vcore_layers",
    "pack_ternary_to_uint8",
    "unpack_uint8_to_ternary",
    "repack_safetensors_layerwise",
    "repack_sharded_model",
    "repack_hf_hub_model",
]
