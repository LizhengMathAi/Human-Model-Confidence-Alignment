import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.resnet import ResNet
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.segmentation.fcn import FCN
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

import transformers
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput, CausalLMOutput, Seq2SeqLMOutput

from typing import Optional, Tuple, List, Union, Callable
from collections import OrderedDict
import types


def trp_criterion(trp_blocks: nn.ModuleList, shared_head: Callable, criterion: Callable, lambdas: List[float], hidden_states: Tensor, logits: Tensor, targets: Tensor, loss_normalization=False):
    loss, mask = criterion(logits, targets)
    if loss_normalization:
        coeff = loss.detach()

    embeds = [hidden_states]
    predictions = []
    for k, c in enumerate(lambdas):
        embeds.append(trp_blocks[k](embeds[-1]))
        predictions.append(shared_head(embeds[-1]))
        replica_loss, mask = criterion(predictions[-1], targets, mask)
        loss += c * replica_loss
    
    if loss_normalization:
        with torch.no_grad():
            coeff = torch.exp(coeff) / torch.exp(loss.detach())
        loss = coeff * loss
    
    return loss


class TPBlock(nn.Module):
    def __init__(self, depths: int, in_features: int, p: float, dim=-1):
        super(TPBlock, self).__init__()

        self.dropout = nn.Dropout(p)

        self.cdim = dim

        blocks = []
        for _ in range(depths):
            blocks.append(nn.Linear(in_features, in_features))
            nn.init.constant_(blocks[-1].weight, 0.0)
            nn.init.constant_(blocks[-1].bias, 0.0)
            blocks.append(nn.ReLU())
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.dropout(x)
        if self.cdim == -1:
            x = x + self.blocks(x)
        else:
            x = x + torch.movedim(self.blocks(torch.movedim(x, self.cdim, -1)), -1, self.cdim)
        return x
    

class Config:
    @staticmethod
    def gen_criterion(*args, **kwargs):
        def func(input, target, mask=None):
            """
            Args:
                input (Tensor): Input tensor.
                target (Tensor): Target labels.

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor with the same shape of target.
            """
            pass
        return func
    
    @staticmethod
    def gen_shared_head(*args, **kwargs):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States tensor.

            Returns:
                logits (Tensor): Logits tensor.
            """
            pass
        return func

    @staticmethod
    def forward(*args, **kwargs):
        pass


# Wav2Vec2 for Audio Classification
class Wav2Vec2ForSequenceClassificationConfig(Config): 
    _HIDDEN_STATES_START_POSITION = 2

    @staticmethod
    def gen_criterion():
        def func(input, target, mask=None):
            """
            Args:
                input (Tensor): Input tensor of shape [B, C].
                target (Tensor): Target labels of shape [B].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            if mask is None:
                mask = torch.ones_like(target, dtype=torch.float32, device=target.device)

            unmasked_loss = F.cross_entropy(input, target, reduction="none")
            loss = torch.sum(mask * unmasked_loss) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                mask = mask * torch.eq(torch.argmax(input, dim=1), target).to(input.dtype)

            return loss, mask
        return func
    
    @staticmethod
    def gen_shared_head(self, attention_mask):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, L, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            _hidden_states = self.projector(hidden_states)
            if attention_mask is None:
                pooled_output = _hidden_states.mean(dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(_hidden_states.shape[1], attention_mask)
                expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, _hidden_states.shape[2])
                _hidden_states[~expand_padding_mask] = 0.0
                pooled_output = _hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

            logits = self.classifier(pooled_output)
            return logits
        return func
    
    @staticmethod
    def gen_forward(lambdas, loss_normalization=False):
        def func(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.config.use_weighted_layer_sum:
                hidden_states = outputs[Wav2Vec2ForSequenceClassificationConfig._HIDDEN_STATES_START_POSITION]
                hidden_states = torch.stack(hidden_states, dim=1)
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            else:
                hidden_states = outputs[0]

            _hidden_states = self.projector(hidden_states)
            if attention_mask is None:
                pooled_output = _hidden_states.mean(dim=1)
            else:
                padding_mask = self._get_feature_vector_attention_mask(_hidden_states.shape[1], attention_mask)
                expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, _hidden_states.shape[2])
                _hidden_states[~expand_padding_mask] = 0.0
                pooled_output = _hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                shared_head = Wav2Vec2ForSequenceClassificationConfig.gen_shared_head(self, attention_mask)
                criterion = Wav2Vec2ForSequenceClassificationConfig.gen_criterion()
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, hidden_states, logits.view(-1, self.config.num_labels),  labels.view(-1), loss_normalization)  # NOTE: Apply TRP!

            if not return_dict:
                output = (logits,) + outputs[Wav2Vec2ForSequenceClassificationConfig._HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return func


# MobileNetV2 for Image Classification
class MobileNetV2Config(Config):
    @staticmethod
    def gen_criterion(label_smoothing=0.0, top_k=1):
        def func(input, target, mask=None):
            """
            Args:
                input (Tensor): Input tensor of shape [B, C].
                target (Tensor): Target labels of shape [B] or [B, C].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            label = torch.argmax(target, dim=1) if label_smoothing > 0.0 else target
                
            unmasked_loss = F.cross_entropy(input, label, reduction="none", label_smoothing=label_smoothing)
            if mask is None:
                mask = torch.ones_like(unmasked_loss, dtype=torch.float32, device=target.device)
            loss = torch.sum(mask * unmasked_loss) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                topk_values, topk_indices = torch.topk(input, top_k, dim=-1)
                mask = mask * torch.eq(topk_indices, label[:, None]).any(dim=-1).to(input.dtype)

            return loss, mask
        return func
    
    @staticmethod
    def gen_shared_head(self):
        def func(x):
            """
            Args:
                x (Tensor): Hidden States tensor of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.classifier(x)
            return logits
        return func

    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, label_smoothing=0.0, top_k=1):
        def func(self, images: Tensor, targets=None):
            x = self.features(images)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            logits = self.classifier(x)

            if self.training:
                torch._assert(targets is not None, "targets should not be none when in training mode")
                shared_head = MobileNetV2Config.gen_shared_head(self)
                criterion = MobileNetV2Config.gen_criterion(label_smoothing, top_k)
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, x, logits, targets, loss_normalization)
                return logits, loss
            return logits
        return func


# ResNet for Image Classification
class ResNetConfig(MobileNetV2Config):
    @staticmethod
    def gen_shared_head(self):
        def func(x):
            """
            Args:
                x (Tensor): Hidden States tensor of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.fc(x)
            return logits
        return func

    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, label_smoothing=0.0, top_k=1):
        def func(self, images: Tensor, targets=None):
            x = self.conv1(images)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.fc(x)

            if self.training:
                torch._assert(targets is not None, "targets should not be none when in training mode")
                shared_head = ResNetConfig.gen_shared_head(self)
                criterion = ResNetConfig.gen_criterion(label_smoothing, top_k)
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, x, logits, targets, loss_normalization)
                return logits, loss
            return logits
        return func


# EfficientNet for Image Classification
class EfficientNetConfig(MobileNetV2Config):
    @staticmethod
    def gen_shared_head(self):
        def func(x):
            """
            Args:
                x (Tensor): Hidden States tensor of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.classifier(x)
            return logits
        return func

    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, label_smoothing=0.0, top_k=1):
        def func(self, images: Tensor, targets=None):
            x = self.features(images)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)

            if self.training:
                torch._assert(targets is not None, "targets should not be none when in training mode")
                shared_head = EfficientNetConfig.gen_shared_head(self)
                criterion = EfficientNetConfig.gen_criterion(label_smoothing, top_k)
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, x, logits, targets, loss_normalization)
                return logits, loss
            return logits
        return func


# ViT for Image Classification
class VisionTransformerConfig(MobileNetV2Config):
    @staticmethod
    def gen_shared_head(self):
        def func(x):
            """
            Args:
                x (Tensor): Hidden States tensor of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.heads(x)
            return logits
        return func

    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, label_smoothing=0.0, top_k=1):
        def func(self, images: Tensor, targets=None):
            x = self._process_input(images)
            n = x.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            x = x[:, 0]

            logits = self.heads(x)

            if self.training:
                torch._assert(targets is not None, "targets should not be none when in training mode")
                shared_head = VisionTransformerConfig.gen_shared_head(self)
                criterion = VisionTransformerConfig.gen_criterion(label_smoothing, top_k)
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, x, logits, targets, loss_normalization)
                return logits, loss
            return logits
        return func


# Bert for Question Answering
class BertForQuestionAnsweringConfig(Config): 
    @staticmethod
    def gen_criterion(top_k=1):
        def func(input, target: List[Tensor], mask=None):
            """
            Args:
                input (Tensor): Input tensor of shape [B, C, 2].
                target (List[Tensor]): 
                    Start Positions of shape [B].
                    End Positions of shape [B].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            start_positions, end_positions = target

            if mask is None:
                mask = torch.ones_like(start_positions, dtype=torch.float32, device=start_positions.device)

            start_logits, end_logits = input.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            masked_start_losses = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index, reduction="none")
            start_loss = torch.sum(mask * masked_start_losses) / (torch.sum(mask) + 1e-6)
            masked_end_losses = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index, reduction="none")
            end_loss = torch.sum(mask * masked_end_losses) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                topk_values, topk_indices = torch.topk(start_logits, top_k, dim=1)
                mask = mask * torch.eq(topk_indices, start_positions[:, None]).any(dim=1).to(start_logits.dtype)
                topk_values, topk_indices = torch.topk(end_logits, top_k, dim=1)
                mask = mask * torch.eq(topk_indices, end_positions[:, None]).any(dim=1).to(end_logits.dtype)

            return (start_loss + end_loss) / 2, mask
        return func
    
    @staticmethod
    def gen_shared_head(self):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, C, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C, 2].
            """
            logits = self.qa_outputs(hidden_states)
            return logits
        return func
    
    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, top_k=1):
        def func(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
            r"""
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            total_loss = None
            if start_positions is not None and end_positions is not None:
                shared_head = BertForQuestionAnsweringConfig.gen_shared_head(self)
                criterion = BertForQuestionAnsweringConfig.gen_criterion()
                total_loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, sequence_output,  logits, [start_positions, end_positions], loss_normalization)  # NOTE: Apply TRP!

            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return func
    

# FCN for Semantic Segmentation
class FCNConfig(Config):
    @staticmethod
    def gen_criterion(top_k=1):
        def func(input, target, mask=None):
            """
            Args:
                input Tensor: input tensor of shape [B, C, H, W].
                target (Tensor): Target labels of shape [B, H, W].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B, H, W].
            """
            if mask is None:
                mask = torch.ones_like(target, dtype=torch.float32, device=target.device)
                
            masked_loss = F.cross_entropy(input, target, ignore_index=255, reduction="none")
            loss = torch.sum(mask * masked_loss) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                topk_values, topk_indices = torch.topk(input, top_k, dim=1)
                mask = mask * torch.eq(topk_indices, target[:, None, :, :]).any(dim=1).to(input.dtype)
                # mask = mask * torch.eq(torch.argmax(x, dim=1), target).to(x.dtype)
                
            return loss, mask
        return func
    
    @staticmethod
    def gen_out_shared_head(self, input_shape):
        def func(features):
            """
            Args:
                features (Tensor): features tensor of shape [B, hidden_units, H, W].

            Returns:
                result (Tensors): result tensor of shape [B, C, H, W].
            """
            x = self.classifier(features)
            result = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            return result
        return func
    
    @staticmethod
    def gen_aux_shared_head(self, input_shape):
        def func(features):
            """
            Args:
                features (Tensor): features tensor of shape [B, hidden_units, H, W].

            Returns:
                result (Tensors): result tensor of shape [B, C, H, W].
            """
            x = self.aux_classifier(features)
            result = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            return result
        return func

    @staticmethod
    def gen_forward(lambdas, loss_normalization=True, top_k=1):
        def func(self, images: Tensor, targets=None):
            input_shape = images.shape[-2:]
            # contract: features is a dict of tensors
            features = self.backbone(images)

            result = OrderedDict()
            x = features["out"]
            x = self.classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["out"] = x

            if self.aux_classifier is not None:
                x = features["aux"]
                x = self.aux_classifier(x)
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
                result["aux"] = x

            if self.training:
                torch._assert(targets is not None, "targets should not be none when in training mode")
                out_shared_head = FCNConfig.gen_out_shared_head(self, input_shape)
                aux_shared_head = FCNConfig.gen_aux_shared_head(self, input_shape)
                criterion = FCNConfig.gen_criterion(top_k)
                out_loss = trp_criterion(self.out_trp_blocks, out_shared_head, criterion, lambdas, features["out"], result["out"], targets, loss_normalization)
                aux_loss = trp_criterion(self.aux_trp_blocks, aux_shared_head, criterion, lambdas, features["aux"], result["aux"], targets, loss_normalization)
                loss = out_loss + 0.5 * aux_loss
                return result, loss
            return result
        return func


# DeepLabV3Config for Semantic Segmentation
class DeepLabV3Config(FCNConfig):
    pass


# Bert for Text Classification
class BertForSequenceClassificationConfig(Config): 
    @staticmethod
    def gen_criterion():
        def func(input, target, mask=None):
            """
            Args:
                input (Tensor): Input tensor of shape [B, C].
                target (Tensor): Target labels of shape [B].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            if mask is None:
                mask = torch.ones_like(target, dtype=torch.float32, device=target.device)

            unmasked_loss = F.cross_entropy(input, target, reduction="none")
            loss = torch.sum(mask * unmasked_loss) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                mask = mask * torch.eq(torch.argmax(input, dim=1), target).to(input.dtype)

            return loss, mask
        return func
    
    @staticmethod
    def gen_shared_head(self):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.classifier(hidden_states)
            return logits
        return func
        
    @staticmethod
    def gen_forward(lambdas, loss_normalization=False):
        def func(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                assert self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int)  # TODO: remove this 
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    if self.num_labels == 1:
                        loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                    else:
                        loss = F.mse_loss(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    shared_head = BertForSequenceClassificationConfig.gen_shared_head(self)
                    criterion = BertForSequenceClassificationConfig.gen_criterion()
                    loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, pooled_output,  logits, labels, loss_normalization)
                elif self.config.problem_type == "multi_label_classification":
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return func


# Boberta for Text Classification
class RobertaForSequenceClassificationConfig(BertForSequenceClassificationConfig): 
    @staticmethod
    def gen_shared_head(self):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C].
            """
            logits = self.classifier(hidden_states)
            return logits
        return func
        
    @staticmethod
    def gen_forward(lambdas, loss_normalization=False):
        def func(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                assert self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int)  # TODO: remove this 
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    if self.num_labels == 1:
                        loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                    else:
                        loss = F.mse_loss(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    shared_head = BertForSequenceClassificationConfig.gen_shared_head(self)
                    criterion = BertForSequenceClassificationConfig.gen_criterion()
                    loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, sequence_output,  logits, labels, loss_normalization)
                elif self.config.problem_type == "multi_label_classification":
                    loss = F.binary_cross_entropy_with_logits(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return func


# Wav2Vec2 for Speech Recognition
class Wav2Vec2ForCTCConfig(Config):
    _HIDDEN_STATES_START_POSITION = 2

    @staticmethod
    def greedy_decode_ctc(
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor,
        blank_token_id: int,
        target_lengths: torch.Tensor
    ):
        """
        Convert logits to flattened predictions that match the shape of flattened_targets.

        Args:
            log_probs: [B, L, V] - log-softmax output
            input_lengths: [B] - actual length of each input
            blank_token_id: int - index of blank token
            target_lengths: [B] - used to determine how many predictions to keep per sample

        Returns:
            flattened_predictions: 1D tensor, same total length as sum(target_lengths)
        """
        batch_size = log_probs.size(0)
        decoded_all = []

        predicted_ids = log_probs.argmax(dim=-1)  # [B, L]

        for i in range(batch_size):
            pred = predicted_ids[i][:input_lengths[i]]  # [Li]
            prev = None
            decoded = []
            for token in pred:
                token = token.item()
                if token != blank_token_id and token != prev:
                    decoded.append(token)
                prev = token
            # Trim or pad to match target_lengths[i]
            tgt_len = target_lengths[i].item()
            if len(decoded) >= tgt_len:
                decoded = decoded[:tgt_len]
            else:
                decoded = decoded + [blank_token_id] * (tgt_len - len(decoded))  # pad with blank
            decoded_all.extend(decoded)

        return torch.tensor(decoded_all, dtype=torch.long, device=log_probs.device)  # shape: [sum(target_lengths)]

    @staticmethod
    def gen_criterion(input_lengths: Tensor, pad_token_id: int, ctc_zero_infinity: bool):
        def func(logits: Tensor, labels: Tensor, mask=None):
            """
            Args:
                logits (Tensor): Log Probablities of shape [B, L, V].
                labels (Tensor): Flattened Targets of shape [B, L'].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            if mask is None:
                mask = torch.ones_like(input_lengths, dtype=torch.float32, device=input_lengths.device)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            with torch.backends.cudnn.flags(enabled=False):
                masked_losses = nn.functional.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=pad_token_id, reduction="none", zero_infinity=ctc_zero_infinity)
                loss = torch.sum(mask * masked_losses) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                thres = 0.5
                flattened_predictions = Wav2Vec2ForCTCConfig.greedy_decode_ctc(
                    log_probs.transpose(0, 1),  # [B, T, V]
                    input_lengths=input_lengths,
                    blank_token_id=pad_token_id,
                    target_lengths=target_lengths
                )
                token_wise_mask =  torch.eq(flattened_predictions, flattened_targets).to(flattened_targets.dtype)
                segment_ids = torch.arange(len(target_lengths), device=target_lengths.device).repeat_interleave(target_lengths)
                sequence_wise_mask = torch.zeros(len(target_lengths), dtype=target_lengths.dtype, device=token_wise_mask.device).scatter_add(0, segment_ids, token_wise_mask)
                mask = mask * torch.ge(sequence_wise_mask, thres * target_lengths).to(flattened_targets.dtype)

            return loss, mask
        return func
    
    @staticmethod
    def gen_shared_head(self):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, C, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, C, 2].
            """
            logits = self.lm_head(hidden_states)
            # log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            return logits
        return func
    
    @staticmethod
    def gen_forward(lambdas, loss_normalization=False):
        def func(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, CausalLMOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
                the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if labels is not None and labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            hidden_states = self.dropout(hidden_states)

            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # retrieve loss input_lengths from attention_mask
                attention_mask = (
                    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
                )
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
                shared_head = Wav2Vec2ForCTCConfig.gen_shared_head(self)
                criterion = Wav2Vec2ForCTCConfig.gen_criterion(input_lengths, self.config.pad_token_id, self.config.ctc_zero_infinity)
                loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, hidden_states,  logits, labels, loss_normalization)  # NOTE: Apply TRP!

            if not return_dict:
                output = (logits,) + outputs[Wav2Vec2ForCTCConfig._HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutput(
                loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )
        return func


# MBart for Translation
class MBartForConditionalGenerationConfig(Config): 
    @staticmethod
    def gen_criterion(vocab_size: int, top_k=1):
        def func(logits, labels, mask=None):
            """
            Args:
                logits (Tensor): Logits tensor of shape [B, L, V].
                labels (Tensor): Target labels of shape [B, L].

            Returns:
                loss (Tensor): Scalar tensor representing the loss.
                mask (Tensor): Boolean mask tensor of shape [B].
            """
            if mask is None:
                mask = torch.ones_like(labels.view(-1), dtype=torch.float32, device=labels.device)
                
            masked_losses = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction="none")
            loss = torch.sum(mask * masked_losses) / (torch.sum(mask) + 1e-6)

            with torch.no_grad():
                topk_values, topk_indices = torch.topk(logits.view(-1, vocab_size), top_k, dim=1)
                mask = mask * torch.eq(topk_indices, labels.view(-1, 1)).any(dim=1).to(logits.dtype)

            return loss, mask
        return func
    
    @staticmethod
    def gen_shared_head(self):
        def func(hidden_states):
            """
            Args:
                hidden_states (Tensor): Hidden States of shape [B, L, hidden_units].

            Returns:
                logits (Tensor): Logits tensor of shape [B, L].
            """
            logits = self.lm_head(hidden_states) + self.final_logits_bias
            return logits
        return func
    
    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
        """
        Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
        have a single `decoder_start_token_id` in contrast to other Bart-like models.
        """
        prev_output_tokens = input_ids.clone()

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

        index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
        prev_output_tokens[:, 0] = decoder_start_tokens

        return prev_output_tokens

    @staticmethod
    def gen_forward(lambdas, loss_normalization=False):
        def func(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            Returns:

            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if labels is not None:
                # if use_cache:
                #     logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
                use_cache = False
                if decoder_input_ids is None and decoder_inputs_embeds is None:
                    decoder_input_ids = MBartForConditionalGenerationConfig.shift_tokens_right(labels, self.config.pad_token_id)

            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

            masked_lm_loss = None
            if labels is not None:
                shared_head = MBartForConditionalGenerationConfig.gen_shared_head(self)
                criterion = MBartForConditionalGenerationConfig.gen_criterion(self.config.vocab_size)
                masked_lm_loss = trp_criterion(self.trp_blocks, shared_head, criterion, lambdas, outputs[0],  lm_logits, labels, loss_normalization)

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        return func


def apply_trp(model, depths: int, p: float, lambdas: List[float], **kwargs):
    if isinstance(model, transformers.Wav2Vec2ForSequenceClassification):
        print("✅ Applying TRP to Wav2Vec2 for Audio Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 768, p) for _ in lambdas])
        model.forward = types.MethodType(Wav2Vec2ForSequenceClassificationConfig.gen_forward(lambdas, False), model)
    elif isinstance(model, MobileNetV2):
        print("✅ Applying TRP to MobileNetV2 for Image Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1280, p) for _ in lambdas])
        model.forward = types.MethodType(MobileNetV2Config.gen_forward(lambdas, True, label_smoothing=kwargs["label_smoothing"], top_k=1), model)
    elif isinstance(model, ResNet):
        print("✅ Applying TRP to ResNet for Image Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 2048, p) for _ in lambdas])
        model.forward = types.MethodType(ResNetConfig.gen_forward(lambdas, True, label_smoothing=kwargs["label_smoothing"], top_k=1), model)
    elif isinstance(model, EfficientNet):
        print("✅ Applying TRP to EfficientNet for Image Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1280, p) for _ in lambdas])
        model.forward = types.MethodType(EfficientNetConfig.gen_forward(lambdas, True, label_smoothing=kwargs["label_smoothing"], top_k=1), model)
    elif isinstance(model, VisionTransformer):
        print("✅ Applying TRP to VisionTransformer for Image Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 768, p) for _ in lambdas])
        model.forward = types.MethodType(VisionTransformerConfig.gen_forward(lambdas, True, label_smoothing=kwargs["label_smoothing"], top_k=1), model)
    elif isinstance(model, transformers.BertForQuestionAnswering):
        print("✅ Applying TRP to Bert for Question Answering...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 768, p) for _ in lambdas])
        model.forward = types.MethodType(BertForQuestionAnsweringConfig.gen_forward(lambdas, True, 1), model)
    elif isinstance(model, FCN):
        print("✅ Applying TRP to FCN for Semantic Segmentation...")
        model.out_trp_blocks = torch.nn.ModuleList([TPBlock(depths, 2048, p, dim=1) for _ in lambdas])
        model.aux_trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1024, p, dim=1) for _ in lambdas])
        model.forward = types.MethodType(FCNConfig.gen_forward(lambdas, True, 1), model)
    elif isinstance(model, DeepLabV3):
        print("✅ Applying TRP to DeepLabV3 for Semantic Segmentation...")
        model.out_trp_blocks = torch.nn.ModuleList([TPBlock(depths, 2048, p, dim=1) for _ in lambdas])
        model.aux_trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1024, p, dim=1) for _ in lambdas])
        model.forward = types.MethodType(DeepLabV3Config.gen_forward(lambdas, True, 1), model)
    elif isinstance(model, transformers.BertForSequenceClassification):
        print("✅ Applying TRP to Bert for Text Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 768, p) for _ in lambdas])
        model.forward = types.MethodType(BertForSequenceClassificationConfig.gen_forward(lambdas, False), model)
    elif isinstance(model, transformers.RobertaForSequenceClassification):
        print("✅ Applying TRP to Roberta for Text Classification...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 768, p) for _ in lambdas])
        model.forward = types.MethodType(RobertaForSequenceClassificationConfig.gen_forward(lambdas, False), model)
    elif isinstance(model, transformers.Wav2Vec2ForCTC):
        print("✅ Applying TRP to Wav2Vec2 for Speech Recognition...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1024, p) for _ in lambdas])
        model.forward = types.MethodType(Wav2Vec2ForCTCConfig.gen_forward(lambdas, False), model)
    elif isinstance(model, transformers.MBartForConditionalGeneration):
        print("✅ Applying TRP to MBart for Translation...")
        model.trp_blocks = torch.nn.ModuleList([TPBlock(depths, 1024, p) for _ in lambdas])
        model.forward = types.MethodType(MBartForConditionalGenerationConfig.gen_forward(lambdas, False), model)
    else:
        torch._assert(
            isinstance(model, transformers.Wav2Vec2ForSequenceClassification), 
            "The model should be an object of [`Wav2Vec2ForSequenceClassification`].")

    return model