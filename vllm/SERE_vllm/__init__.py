# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry

def register():

    from .sere_qwen2_moe import Qwen2MoeForCausalLMSERE
    if "Qwen2MoeForCausalLMSERE" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2MoeForCausalLMSERE", Qwen2MoeForCausalLMSERE)

    from .sere_deepseek_v2 import DeepseekV2ForCausalLMSERE
    if "DeepseekV2ForCausalLMSERE" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("DeepseekV2ForCausalLMSERE", DeepseekV2ForCausalLMSERE)

    from .sere_qwen3_moe import Qwen3MoeForCausalLMSERE
    if "Qwen3MoeForCausalLMSERE" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen3MoeForCausalLMSERE", Qwen3MoeForCausalLMSERE)