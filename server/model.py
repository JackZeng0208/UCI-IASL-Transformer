from typing import Optional, Tuple

import accelerate
import torch
import torch.nn as nn
from orin.config import OPTConfig
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)


def build_opt_model(model_size: str, model_name: str = "facebook/opt-") -> Tuple[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]:
    model_size = model_size.lower()
    assert (model_size == "125m" or model_size == "350m" or model_size == "1.3b"
            or model_size == "2.7b" or model_size == "6.7b")
    config = AutoConfig.from_pretrained(
        model_name + model_size, output_hidden_state=True)
    model = AutoModelForCausalLM.from_config(config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name + model_size)
    return model, tokenizer, config
