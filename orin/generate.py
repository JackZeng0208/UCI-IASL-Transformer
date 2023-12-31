from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from orin.config import OPTConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, OPTForCausalLM


def multinomial_sample_one_no_sync(probs_sort):
    # does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    # top_k: default set to 200, refer to top k sampling
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


@torch.no_grad()
def speculative_decoding(
    draft_model: PreTrainedModel,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    speculative_k: int,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    draft_tokens, draft_probs = [0 for _ in range(speculative_k)], [0.0 for _ in range(speculative_k)]
    draft_model.to(OPTConfig.device)
    draft_outputs = input_ids.clone().to(OPTConfig.device)
    for i in range(speculative_k):
        draft_outputs = dict(draft_model.generate(
            draft_outputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        ))
        sequences = draft_outputs["sequences"]
        scores = draft_outputs["scores"][0]
        draft_probs[i] = logits_to_probs(scores, top_k=top_k)
        draft_tokens[i] = sequences[0][-1]

    # validate draft model outputs by feeding draft tokens with inputs into target model 
    # and compare the prediction on the next token
    target_outputs = dict(model.generate(
        torch.cat([input_ids.clone().to(OPTConfig.device), draft_tokens]),
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
    ))

    target_probs = logits_to_probs(target_outputs["scores"][0], top_k=top_k)
    target_tokens = target_outputs["sequences"]
    draft_probs = torch.stack(draft_probs)
    # p: draft prob, q: target prob
    p = draft_probs[torch.arange(0, speculative_k, device=OPTConfig.device), draft_tokens]
    q = target_probs[torch.arange(0, speculative_k, device=OPTConfig.device), draft_tokens]

    # get probs for each speculate token
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculative_k] / p)
    # determine whether to accept each speculate token based on its accept prob
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:
        # accept all draft tokens
        last_token = target_tokens[-1]
        return torch.cat([draft_tokens, last_token])
    else:
        # get the first rejection location
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: torch.Tensor,
    max_new_tokens: int,
):
    model.to(OPTConfig.device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = input_ids.to(OPTConfig.device)
    for _ in range(max_new_tokens):
        outputs = model.generate(outputs, max_new_tokens=1, do_sample=False)
        next_token = outputs[0][-1]
        # if generated token is end of sentence token, stop the inference
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:])
    return generated_text
