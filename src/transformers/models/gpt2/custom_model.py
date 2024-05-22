import torch
from transformers import AutoTokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithCrossAttentions


from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate import load
import argparse
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class GPT2WithThresholdedAttention(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # ---- THESE LINES ARE IDENTICAL TO GPT2LMHEADMODEL's Forward -----
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # ---- END OF LINES THAT ARE IDENTICAL TO GPT2LMHEADMODEL's Forward -----
        # MAIN modification: Modify attention
        attentions = list(transformer_outputs.attentions)
        new_attentions = self.modified_attention(attentions, self.alpha)
        for i, layer_attention in enumerate(new_attentions):
            attentions[i] = layer_attention
        # Add modified attentions to the output
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    '''
    Creates new_attentions from attentions = Tuple of tensor (batch_size, num_heads, seq_length, seq_length)

    In this new attention_weights, we identify the greatest set of indices for each row in attentions whose sum is less than the threshold.
    We zero out remaining elements.

    Threshold is defined as alpha * total_sum, where total_sum is the sum over axis 3 in the attention matrix.
    Thus alpha = 0 accepts nothing, alpha = 1 everything.

    new_attentions has same size as attentions = Tuple of tensor (batch_size, num_heads, seq_length, seq_length)
    '''
    def modified_attention(self, attentions, alpha):
        new_attentions = []
        # Iterate over each layer
        for layer_idx, attention in enumerate(attentions):
            # Each attention is of size (batch_size, num_heads, seq_length, seq_length)
            sorted_weights, indices = torch.sort(attention, dim=3, descending=True)  # Sorted along last dimension
            cumulative_sum = torch.cumsum(sorted_weights, dim=3)  # Cumulative sum of sorted weights
            total_sum = cumulative_sum[:, :, :, -1].unsqueeze(dim=3)  # Total sum along the last dimension

            # Create a mask from sorted weights where cumulative sum is less than alpha * total_sum
            mask_sorted = cumulative_sum < (alpha * total_sum)
            
            # Reorder mask_sorted back to the original attention shape
            mask = torch.zeros_like(mask_sorted)
            mask.scatter_(dim=3, index=indices, src=mask_sorted)

            # Apply mask to the original attention weights
            new_attention = attention * mask.float()  # Convert mask to float for multiplication
            new_attention = new_attention / new_attention.sum(dim=3, keepdim=True) # Renormalize
            new_attentions.append(new_attention)

        return new_attentions