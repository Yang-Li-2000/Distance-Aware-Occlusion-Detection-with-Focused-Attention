# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from magic_numbers import *
import temp_vars


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_decoder_layer_distance=3,
                 num_decoder_layer_occlusion=3,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, d_model=d_model, pair_detector=True)
        if CASCADE:
            # Decoder for distance
            distance_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                             dropout, activation, normalize_before)
            distance_decoder_norm = nn.LayerNorm(d_model)
            self.distance_decoder = TransformerDecoder(distance_decoder_layer, num_decoder_layer_distance, distance_decoder_norm,
                                                       return_intermediate=return_intermediate_dec)
            # Decoder for occlusion
            occlusion_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                              dropout, activation, normalize_before)
            occlusion_decoder_norm = nn.LayerNorm(d_model)
            self.occlusion_decoder = TransformerDecoder(occlusion_decoder_layer, num_decoder_layer_occlusion, occlusion_decoder_norm,
                                                        return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, writer=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        """
        h = ceil(H/32)
        w = ceil(W/32)
        
        query_embed is nn.Embedding().weight
        
        src:                [BS, 256, h, w]
        pos_embed:          [BS, 256, h, w]
        query_embed:        [100, 256]
        mask:               [BS, h, w]
        """
        # Flatten visual features
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        """
        src:                [h*w, BS, 256]
        pos_embed:          [h*w, BS, 256]
        query_embed:        [100, BS, 256]
        mask:               [BS, h*w]
        """

        # Encoder
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        """
        tgt:                [100, BS, 256]
        memory:             [h*w, BS, 256]
        """

        # Decoder
        if VISUALIZE_ATTENTION_WEIGHTS:
            temp_vars.current_decoder = 'pair'
        if IMPROVE_INTERMEDIATE_LAYERS:
            hs, human_outputs_coord, object_outputs_coord = \
                self.decoder(tgt, memory, memory_key_padding_mask=mask,
                             pos=pos_embed, query_pos=query_embed,
                             writer=writer,
                             shape=(bs, c, h, w))
        else:
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed,
                              writer=writer,
                              shape=(bs, c, h, w))
        if not CASCADE:
            if IMPROVE_INTERMEDIATE_LAYERS:
                return hs.transpose(1,2), human_outputs_coord, object_outputs_coord, memory.permute(1, 2, 0).view(bs, c, h, w)
            else:
                return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            hs = hs.transpose(1, 2)

            # Distance
            if VISUALIZE_ATTENTION_WEIGHTS:
                temp_vars.current_decoder = 'dist'
            distance_query_embed = hs[-1]
            distance_query_embed = distance_query_embed.permute(1, 0, 2)
            distance_tgt = torch.zeros_like(distance_query_embed)
            if IMPROVE_INTERMEDIATE_LAYERS:
                distance_decoder_out, _, _ = self.distance_decoder(distance_tgt,
                                                                   memory,
                                                                   memory_key_padding_mask=mask,
                                                                   pos=pos_embed,
                                                                   query_pos=distance_query_embed,
                                                                   shape=(bs, c, h, w))
            else:
                distance_decoder_out = self.distance_decoder(distance_tgt,
                                                             memory,
                                                             memory_key_padding_mask=mask,
                                                             pos=pos_embed,
                                                             query_pos=distance_query_embed,
                                                             shape=(bs, c, h, w))
            distance_decoder_out = distance_decoder_out.transpose(1, 2)

            # Occlusion
            if VISUALIZE_ATTENTION_WEIGHTS:
                temp_vars.current_decoder = 'occl'
            occlusion_query_embed = hs[-1]
            occlusion_query_embed = occlusion_query_embed.permute(1, 0, 2)
            occlusion_tgt = torch.zeros_like(occlusion_query_embed)
            if IMPROVE_INTERMEDIATE_LAYERS:
                occlusion_decoder_out, _, _ = self.occlusion_decoder(
                    occlusion_tgt, memory, memory_key_padding_mask=mask,
                    pos=pos_embed, query_pos=occlusion_query_embed,shape=(bs, c, h, w))
            else:
                occlusion_decoder_out = self.occlusion_decoder(occlusion_tgt,
                                                               memory,
                                                               memory_key_padding_mask=mask,
                                                               pos=pos_embed,
                                                               query_pos=occlusion_query_embed,shape=(bs, c, h, w))
            occlusion_decoder_out = occlusion_decoder_out.transpose(1, 2)

            if IMPROVE_INTERMEDIATE_LAYERS:
                return hs, distance_decoder_out, occlusion_decoder_out, human_outputs_coord, object_outputs_coord, memory.permute(
                    1, 2, 0).view(bs, c, h, w)
            else:
                return hs, distance_decoder_out, occlusion_decoder_out, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class temp_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, pair_detector=False, d_model=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        if IMPROVE_INTERMEDIATE_LAYERS:
            self.pair_detector = pair_detector
            if pair_detector:
                self.human_box_embed = temp_MLP(d_model, d_model, 4, 3)
                self.object_box_embed = temp_MLP(d_model, d_model, 4, 3)
                self.query_pos_proj = torch.nn.Linear(8, d_model)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                writer=None,
                shape=None):
        output = tgt

        intermediate = []
        if IMPROVE_INTERMEDIATE_LAYERS:
            human_outputs_coord_list = []
            object_outputs_coord_list = []

        layer_index = 0
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           writer=writer,
                           shape=shape,
                           layer_index=layer_index)

            if IMPROVE_INTERMEDIATE_LAYERS and self.pair_detector:
                human_outputs_coord = self.human_box_embed(
                    self.norm(output)).sigmoid()
                object_outputs_coord = self.object_box_embed(
                    self.norm(output)).sigmoid()
                query_pos = torch.cat(
                    [human_outputs_coord, object_outputs_coord], dim=2)
                query_pos = self.query_pos_proj(query_pos)
                human_outputs_coord_list.append(human_outputs_coord)
                object_outputs_coord_list.append(object_outputs_coord)


            layer_index += 1
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if IMPROVE_INTERMEDIATE_LAYERS:
            if self.return_intermediate:
                if self.pair_detector:
                    return torch.stack(
                        intermediate), torch.stack(
                        human_outputs_coord_list), torch.stack(
                        object_outputs_coord_list)
                else:
                    return torch.stack(intermediate), None, None
        else:
            if self.return_intermediate:
                return torch.stack(intermediate)

        if IMPROVE_INTERMEDIATE_LAYERS:
            return output.unsqueeze(0), None, None
        else:
            return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # Add position encodings to flattened visual features
        q = k = self.with_pos_embed(src, pos)
        """
        By default, 
        src_mask is None, 
        src_key_padding_mask is [BS, h*w]
        """
        # Self-Attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        """
        By default, self.normalize_before is False,
        so forward_post will be used
        """
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        """
        By default, 
        d_model = 256, 
        nhead = 8, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        activation = 'relu'
        normalize_before = False
        """
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     writer=None,
                     shape=None,
                     layer_index=-1):
        """
        """
        """
        tgt:            [100, 2, 256]
        query_pos       [100, 2, 256]
        
        Initially, q = k = query_pos = nn.Embedding().weight
        """
        # Self-Attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        """
        In the first decoder layer, tgt was made of zeros before summing with tgt2.
        Therefore, 
        tgt = self.norm1(self.dropout1(tgt2)) = outputs of self.self_attn().
        
        In other words, tgt is obtained through 
        self_attention --> dropout --> layer_norm (first layer)
        OR
        self_attention --> dropout --> add to previoius tgt --> layer_norm (subsequent layers)
        """

        # Multi-Head Attention
        tgt2, attention_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0:2]
        if VISUALIZE_ATTENTION_WEIGHTS:
            bs, c, h, w = shape

            # Reshape flatten attention weights to h * w
            assert attention_weights.shape[0] == 1
            attention_weights = attention_weights.detach()          # detach
            attention_weights = attention_weights.squeeze(0)        # squeeze
            attention_weights = attention_weights.view(-1, h, w)    # reshape

            # Store attention weights
            if temp_vars.current_decoder == 'pair':
                temp_vars.attention_pair_decoder.append(attention_weights)
            elif temp_vars.current_decoder == 'dist':
                temp_vars.attention_distance_decoder.append(attention_weights)
            elif temp_vars.current_decoder == 'occl':
                temp_vars.attention_occlusion_decoder.append(attention_weights)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        """
        tgt above is obtained through
        multi_head_attention --> dropout --> add to previous tgt --> layer_norm
        """

        # Feed Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        """
        tgt above is obtained through
        linear --> activation --> dropout --> linear --> dropout 
        --> add to previous tgt --> layer_norm
        """


        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                writer=None,
                shape=None,
                layer_index=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        """
        By default, self.normalize_before is False,
        so self.forward_post will be used
        """
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, writer, shape, layer_index)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        num_decoder_layer_distance=args.dec_layers_distance,
        num_decoder_layer_occlusion=args.dec_layers_occlusion,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
