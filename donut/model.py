import torch
import torch.nn as nn
from typing import Optional
from transformers import VisionEncoderDecoderModel


# copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py
def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss

def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # no need to shift labels
    # VisionEncoderDecoderModel shifts to right before feeding to forward method
    # https://github.com/huggingface/transformers/blob/37152f84464dea9086dd1d88cd58f63c2129ee69/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L412

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = fixed_cross_entropy(logits, labels, num_items_in_batch)
    return loss

class DonutModel(VisionEncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = ForCausalLMLoss

# https://github.com/huggingface/transformers/blob/071eb5334f5a9ac2c7a13515219be8a272388ec6/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L438C1-L443C14
class DonutVQAModel(DonutModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.prompt_end_token_id = None
        causal_lm_loss = self.loss_function

        def vqa_loss_function(logits, labels, vocab_size: int, num_items_in_batch: Optional[torch.Tensor] = None):
                """ignores all losses before and including the prompt_end_token_id"""
                # Get the indices of the prompt_end_token_id for each row
                prompt_end_token_id = self.config.decoder_start_token_id if self.config.prompt_end_token_id is None else self.config.prompt_end_token_id
                prompt_end_indices = torch.argmax((labels == prompt_end_token_id).int(), dim=1)
                # Iterate through each row and apply the masking
                for i in range(labels.shape[0]):
                    labels[i, :prompt_end_indices[i]+1] = -100
                return causal_lm_loss(logits, labels, vocab_size, num_items_in_batch)
        self.loss_function = vqa_loss_function