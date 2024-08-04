import torch as t
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# Custom SelfAttention to simulate Grouped Query Attention
class CustomSelfAttention(t.nn.Module):
    def __init__(self, config, num_attention_heads: Optional[int] = 8):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.all_head_size = config.hidden_size
        self.attention_head_size = int(self.all_head_size / self.num_attention_heads)

        self.query = t.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = t.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = t.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = t.nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Implement the forward pass here
        return hidden_states, None  # Dummy return, replace with actual outputs


class CustomLayer(t.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomSelfAttention(config)
        self.intermediate = t.nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = t.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = t.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = t.nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)

        return (layer_output,)


class CustomEncoder(t.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = t.nn.ModuleList(
            [CustomLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CustomPreTrainedModel(PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        model_class: type,
        encoder_class: Optional[type] = CustomEncoder,
    ):
        super().__init__(config)
        self.model = model_class(config)
        self.encoder = encoder_class(config)
