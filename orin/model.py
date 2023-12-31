# code for model architecture with kv cache module for inference optimization

from typing import Optional, Tuple

import accelerate
import torch
import torch.nn as nn
from orin.config import OPTConfig
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)


class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_length: int, n_heads: int, head_dim: int, dtype=OPTConfig.dtype):
        super().__init__()
        # note: attention shape (batch size, sequence length, hidden dimension) or (B, S, D)
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class EfficientModel(PreTrainedModel):
    # TODO: construct kv cache wrapper to equip existing model with kv cache
    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config

    def setup_caches(self, max_seq_length: int):
        if self.config.max_length >= max_seq_length:
            return


# FIXME: cannot use flash attention for now, only support CUDA version>11.6 (we are on 11.4)
def build_opt_model(model_size: str, model_name: str = "facebook/opt-") -> Tuple[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]:
    model_size = model_size.lower()
    assert (model_size == "125m" or model_size == "350m" or model_size == "1.3b"
            or model_size == "2.7b" or model_size == "6.7b")
    config = AutoConfig.from_pretrained(
        model_name + model_size, output_hidden_state=True)
    model = AutoModelForCausalLM.from_config(config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name + model_size)
    return model, tokenizer, config
