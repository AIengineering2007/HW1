# hw1-build: caspian-7
from typing import Optional
import torch
import torch.nn.functional as F
def generate(
    model,
    prompt_ids: list,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_cache: bool = True,
    seed: Optional[int] = None
) -> list:

    if len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if temperature < 0:
        raise ValueError("Temperature cannot be negative")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    block_size = model.config.block_size

    if len(prompt_ids) + max_new_tokens > block_size:
        raise ValueError("Sequence length exceeds block_size")

    if max_new_tokens == 0:
        return prompt_ids.copy()

    device = next(model.parameters()).device

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()

    result = prompt_ids.copy()
    past_kv = None

    try:
        with torch.no_grad():

            for step in range(max_new_tokens):

                if use_cache:
                    if step == 0:
                        model_input = idx
                    else:
                        model_input = next_token

                    logits, _, past_kv = model(
                        model_input,
                        past_kv=past_kv,
                        use_cache=True
                    )

                else:
                    model_input = torch.tensor(
                        [result],
                        dtype=torch.long,
                        device=device
                    )

                    logits, _ = model(model_input)

                logits = logits[:, -1, :]

                if temperature == 0:
                    next_token = torch.argmax(
                        logits,
                        dim=-1,
                        keepdim=True
                    )

                else:
                    logits = logits / temperature

                    # Apply top-k first (keep exactly k tokens, even with ties)
                    if top_k is not None:
                        k = min(top_k, logits.size(-1))

                        _, top_idx = torch.topk(logits, k, dim=-1)

                        keep = torch.zeros_like(logits, dtype=torch.bool)
                        keep.scatter_(1, top_idx, True)

                        logits = logits.masked_fill(
                            ~keep,
                            float("-inf")
                        )

                    if top_p is not None and top_p < 1.0:

                        sorted_logits, sorted_indices = torch.sort(
                            logits,
                            descending=True,
                            dim=-1
                        )

                        sorted_probs = F.softmax(
                            sorted_logits,
                            dim=-1
                        )

                        cumulative_probs = torch.cumsum(
                            sorted_probs,
                            dim=-1
                        )

                        # Smallest set whose total probability is >= top_p:
                        # drop a token once the mass BEFORE it already reaches top_p.
                        # The first token has 0 mass before it, so it is always kept.
                        sorted_indices_to_remove = (
                            cumulative_probs - sorted_probs
                        ) >= top_p

                        indices_to_remove = torch.zeros_like(
                            logits,
                            dtype=torch.bool
                        )

                        indices_to_remove.scatter_(
                            1,
                            sorted_indices,
                            sorted_indices_to_remove
                        )

                        logits = logits.masked_fill(
                            indices_to_remove,
                            float("-inf")
                        )

                    probs = F.softmax(logits, dim=-1)

                    next_token = torch.multinomial(
                        probs,
                        num_samples=1,
                        generator=generator
                    )

                token_id = next_token.item()
                result.append(token_id)

    finally:
        model.train(was_training)

    return result
