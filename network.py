from typing import List, Tuple, Union
import math
import torch
from torch import nn
from torch.nn import functional as F

# import commons
# import modules
# import attentions
import monotonic_align
from transforms import piecewise_rational_quadratic_transform

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# from commons import utils.init_weights, utils.get_padding

import utils


class StochasticDurationPredictor(nn.Module):
    __doc__ = r"""Stochastic duration predictor.

    Predict duration of each character(phoneme).
    Use Flow-based generative model for prediction.

    Args:
        in_channels(int): n channels of input.
        filter_channels(int): hidden channels.
        kernel_size(int): kernel size of ``DDSConv`` and ``ConvFlow``
        dropout_ratio(float): dropout ratio.
        n_flows(int): number of layers.
        gin_channels(int): dim of condition.
    """

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        dropout_ratio: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, dropout_ratio=dropout_ratio)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, dropout_ratio=dropout_ratio)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask
        # x는 text로 일종의 condition이라고 할 수 있음.

        if not reverse:
            flows = self.flows
            assert w is not None
            # w = attn.sum(2) -> 각 phoneme의 duration값들이라고 할 수 있음.

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class ConvFlow(nn.Module):
    __doc__ = r"""Used in StochasticDurationPredictor"""

    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: Union[float, int] = 5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, dropout_ratio=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask  # [b, 1, text_len]

        b, c, t = x0.shape  # [batch_size, 2, text_len]
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]
        # [b, 2, 1, t]
        # [b, 2, t, 1]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)  # num_bins
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)  # num_bins
        unnormalized_derivatives = h[..., 2 * self.num_bins :]  # num_bins*2

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class DurationPredictor(nn.Module):
    __doc__ = r"""Duration predictor.

    Predict duration of each character(phoneme).
    It doesn't use Flow-based generative model.
    So prediction of duration is deterministic.
    But faster than ``StochasticDurationPredictor``

    Args:
        in_channels: n channels of input.
        filter_channels: hidden channels.
        kernel_size: kernel size of conv layers.
        dropout_ratio: dropout ratio.
        gin_channels: dim of condition.
    """

    def __init__(
        self, in_channels: int, filter_channels: int, kernel_size: int, dropout_ratio: float, gin_channels: int = 0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio
        self.gin_channels = gin_channels

        self.dropout = nn.Dropout(dropout_ratio)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.dropout(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.dropout(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    __doc__ = r"""Multi head attention in transformer.

    It uses relative positional encoding.
    Paper: <https://arxiv.org/abs/1803.02155>

    Args:
        channels: dimension of input tensor.
        out_channels: dimension of output tensor.
        n_heads: number of heads.
        dropout_ratio: dropout ratio.
        window_size: window size used in relative positional encoding.
        heads_share: whether to share the heads.
        block_length: 
        proximal_bias: 
        proximal_init: whether to share the weights of ``conv_q`` and ``conv_k``.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        dropout_ratio: float = 0.0,
        window_size: int = None,
        heads_share: bool = True,
        block_length: int = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init  # query and key share initial values of weights and biases
        self.attn = None

        self.head_dim = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_ratio)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.head_dim**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.head_dim) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.head_dim) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        # b: batch_size, d: hidden_dim, t_s: key_len, t_t: query_len

        # split heads
        query = query.view(b, self.n_heads, self.head_dim, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.head_dim, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.head_dim, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.head_dim), key.transpose(-2, -1))

        # relative attention
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.head_dim), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        # max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings, utils.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]])
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, utils.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, utils.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(x, utils.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, utils.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
            length: an integer scalar.
        Returns:
            a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class LayerNorm(nn.Module):
    __doc__ = r"""Layer normalization used in transformer encoder"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class FFN(nn.Module):
    __doc__ = r"""Feed forward network for transformer encoder.

    Use ``Conv1d`` not ``Linear``.

    Args:
        in_channels: dimension of input tensor.
        out_channels: dimension of output tensor.
        filter_channels: hidden channels.
        kernel_size: kernel size of conv layers.
        dropout_ratio: dropout ratio.
        activation: whether to use gelu activation or relu.
        causal: whether to use causal padding or same padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        dropout_ratio: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, utils.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, utils.convert_pad_shape(padding))
        return x


class TransformerEncoder(nn.Module):
    __doc__ = r"""Transformer encoder.

    (MultiHeadAttention - LayerNorm - FFN - LayerNorm) * n_layers.

    Args:
        hidden_channels: hidden channels of all layers.
        filter_channels: hidden channels of FFN.
        n_heads: number of heads.
        n_layers: number of layers.
        kernel_size: kernel size of conv layer in FFN.
        dropout_ratio: dropout ratio.
        window_size: window size used for relative positional encoding in MultiHeadAttention.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        dropout_ratio: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio
        self.window_size = window_size

        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, dropout_ratio=dropout_ratio, window_size=window_size
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, dropout_ratio=dropout_ratio)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    __doc__ = r"""TextEncoder.

    Args:
        n_vocab (int): number of vocabulary(symbols).
        out_channels (int): dim of z. (= dim of z_theta = dim of TextEncoder output = dim of PosteriorEncoder output)
        hidden_channels (int): hidden_channels of all networks.
        filter_channels (int): filter_channels(hidden_channels) of FFN in TransformerEncoder.
        n_heads (int): n_heads of TransformerEncoder.
        n_layers (int): n_layers of TransformerEncoder.
        kernel_size (int): kernel_size of FFN in TransformerEncoder.
        dropout_ratio (float): dropout_ratio.
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout_ratio: float,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        # # 이렇게하면 모델 이닛에서 영향을 받지 않나? 아니면 모델 이닛 다시한번 normal로 이닛을 하는 형태인가? 어쩌면 model단에서는 init자체를 안할지도?

        self.encoder = TransformerEncoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, dropout_ratio
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, text, text_lengths):
        text = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h] = [batch_size, seq_len, emb_dim]
        text = torch.transpose(text, 1, -1)  # [b, h, t] = [batch_size, emb_dim, seq_len]
        text_mask = torch.unsqueeze(utils.make_mask(text_lengths, text.size(2)), 1).to(text.dtype)

        text = self.encoder(text * text_mask, text_mask)  # [batch_size, hid_ch, seq_len]
        stats = self.proj(text) * text_mask

        mean_text, logs_text = torch.split(stats, self.out_channels, dim=1)  # [batch_size, out_ch, seq_len]
        return text, mean_text, logs_text, text_mask


class ResidualCouplingBlock(nn.Module):
    __doc__ = r"""Stack of ResidualCouplingLayer and Flip.

    ``Flip`` is needed for mixing data.
    z -> z_theta
    if ``reverse=True`` in ``forward`` z_theta -> z

    Args:
        channels (int): dim of z. (= dim of z_theta = dim of TextEncoder output = dim of PosteriorEncoder output)
        hidden_channels (int): hidden_channels of all networks.
        kernel_size (int): kernel size of WN.
        dilation_rate (int): dilation rate of WN.
        n_layers (int): n_layers of WN.
        n_flows (int): n_stack of flow(ResidualCouplingLayer, Flip).
        gin_channels (int): dim of speaker embedding. Default: 0.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    __doc__ = r"""ResidualCouplingLayer

    Used in ResidualCouplingBlock

    Args:
        channels: dimension of input tensor.
        hidden_channels: hidden channels.
        kernel_size: kernel size of conv layer in ``WN``.
        dilation_rate: dilation rate used in ``WN``.
        n_layers: number of layers in ``WN``.
        dropout_ratio: dropout ratio.
        gin_channels: condition embedding channels.
        mean_only: whether to ignore std.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        dropout_ratio: float = 0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            dropout_ratio=dropout_ratio,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class PosteriorEncoder(nn.Module):
    __doc__ = r"""PosteriorEncoder. (Encoder for linear spectrogram)

    Args:
        in_channels (int): n_channels of linear spectrogram.
        out_channels (int): dim of z. (= dim of z_theta = dim of TextEncoder output = dim of PosteriorEncoder output)
        hidden_channels (int): hidden_channels of all networks.
        kernel_size (int): kernel_size of WN.
        dilation_rate (int): dilation rate of WN.
        n_layers (int): number of conv layers in WN.
        gin_channels (int): dim of speaker embedding. Defalt: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, linspec, linspec_lengths, g=None):
        linspec_mask = torch.unsqueeze(utils.make_mask(linspec_lengths, linspec.size(2)), 1).to(linspec.dtype)
        linspec = self.pre(linspec) * linspec_mask
        linspec = self.enc(linspec, linspec_mask, g=g)
        stats = self.proj(linspec) * linspec_mask
        mean_linspec, logs_linspec = torch.split(stats, self.out_channels, dim=1)
        z = (mean_linspec + torch.randn_like(mean_linspec) * torch.exp(logs_linspec)) * linspec_mask
        return z, mean_linspec, logs_linspec, linspec_mask


class ResBlock1(nn.Module):
    __doc__ = r"""Stack of convolutional layers used in ``Generator``."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=utils.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=utils.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=utils.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(utils.init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=utils.get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=utils.get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=utils.get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(utils.init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    __doc__ = r"""Stack of convolutional layers used in ``Generator``."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=utils.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=utils.get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(utils.init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


# class ConvReluNorm(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, dropout_ratio):
#         super().__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.n_layers = n_layers
#         self.dropout_ratio = dropout_ratio
#         assert n_layers > 1, "Number of layers should be larger than 0."

#         self.conv_layers = nn.ModuleList()
#         self.norm_layers = nn.ModuleList()
#         self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
#         self.norm_layers.append(LayerNorm(hidden_channels))
#         self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(dropout_ratio))
#         for _ in range(n_layers - 1):
#             self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
#             self.norm_layers.append(LayerNorm(hidden_channels))
#         self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
#         self.proj.weight.data.zero_()
#         self.proj.bias.data.zero_()

#     def forward(self, x, x_mask):
#         x_org = x
#         for i in range(self.n_layers):
#             x = self.conv_layers[i](x * x_mask)
#             x = self.norm_layers[i](x)
#             x = self.relu_drop(x)
#         x = x_org + self.proj(x)
#         return x * x_mask


class DDSConv(nn.Module):
    __doc__ = r"""Dialted and Depth-Separable Convolution"""

    def __init__(self, channels: int, kernel_size: int, n_layers: int, dropout_ratio: float = 0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio

        self.dropout = nn.Dropout(dropout_ratio)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.dropout(y)
            x = x + y
        return x * x_mask


class WN(nn.Module):
    __doc__ = r"""WN layer.

    Used in PosteriorEncoder and ResidualCouplingBlock(ResidualCouplingLayer).
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        dropout_ratio: int = 0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.dropout_ratio = dropout_ratio

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout_ratio)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)  # [b, 2h * n_layers, seq_len]

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]  # [b, 2h, 1]
            else:
                g_l = torch.zeros_like(x_in)

            acts = utils.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)  # [b, h, seq_len]
            acts = self.dropout(acts)

            res_skip_acts = self.res_skip_layers[i](acts)  # [b, 2h, seq_len]
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask  # [b, h, seq_len]
                output = output + res_skip_acts[:, self.hidden_channels :, :]  # [b, h, seq_len]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class Log(nn.Module):
    __doc__ = r"""Log layer. Calculate logdet and can reverse log(exp)."""

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    __doc__ = r"""Flip channels for mixing channels in flow."""

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    __doc__ = r"""Elementwise affine layer"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class Generator(nn.Module):
    __doc__ = r"""Generator of HiFi-GAN.

    It is used as decoder of VAE. z -> wav.

    Args:
        initial_channel (int): dim of z. (= dim of z_theta = dim of TextEncoder output = dim of PosteriorEncoder output)
        resblock (str): option of decoder(HiFi-GAN generator) resblock. ['1', '2'].
        resblock_kernel_sizes (List[int]): resblock_kernel_sizes of decoder(HiFi-GAN generator).
        resblock_dilation_sizes (List[int]): resblock_dilation_sizes of decoder(HiFi-GAN generator).
        upsample_rates (List[int]): upsample_rates of decoder(HiFi-GAN generator).
        upsample_initial_channel (int): upsample_initial_channel of decoder(HiFi-GAN generator).
        upsample_kernel_sizes (List[int]): upsample_kernel_sizes of decoder(HiFi-GAN generator).
        gin_channels (int): dim of speaker embedding. Default: 0.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[int],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(utils.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(nn.Module):
    __doc__ = r"""MPD(Multi Period Discriminator) of HiFi-GAN"""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(utils.get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(utils.get_padding(kernel_size, 1), 0))),
                norm_f(
                    Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(utils.get_padding(kernel_size, 1), 0))
                ),
                norm_f(
                    Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(utils.get_padding(kernel_size, 1), 0))
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(utils.get_padding(kernel_size, 1), 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    __doc__ = r"""MSD(Multi Scale Discriminator) of HiFi-GAN."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    __doc__ = r"""Discriminator of VITS.

    Discriminate real wav and generated wav.
    MultiPeriodDiscriminator consists of one DiscriminatorS and five DiscriminatorP.
    Each of DiscriminatorP has ``[2, 3, 5, 7, 11]`` as period.

    Args:
        use_spectral_norm (bool): if ``True`` use spectral_norm else weight_norm. Default: False.
            <https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html>
            <https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html>
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    __doc__ = r"""Full network of VITS except Discriminator.

    Args:
        n_vocab (int): number of vocabulary(symbols).
        spec_channels (int): n_channels of linear spectrogram.
        segment_size (int): length of slice for calculating mel_loss.
        inter_channels (int): dim of z. (= dim of z_theta = dim of TextEncoder output = dim of PosteriorEncoder output)
        hidden_channels (int): hidden_channels of all networks.
        filter_channels (int): filter_channels(hidden_channels) of FFN in TransformerEncoder.
        n_heads (int): n_heads of TransformerEncoder.
        n_layers (int): n_layers of TransformerEncoder.
        kernel_size (int): kernel_size of FFN in TransformerEncoder.
        dropout_ratio (float): dropout_ratio.
        resblock (str): option of decoder(HiFi-GAN generator) resblock. ['1', '2'].
        resblock_kernel_sizes (List[int]): resblock_kernel_sizes of decoder(HiFi-GAN generator).
        resblock_dilation_sizes (List[int]): resblock_dilation_sizes of decoder(HiFi-GAN generator).
        upsample_rates (List[int]): upsample_rates of decoder(HiFi-GAN generator).
        upsample_initial_channel (int): upsample_initial_channel of decoder(HiFi-GAN generator).
        upsample_kernel_sizes (List[int]): upsample_kernel_sizes of decoder(HiFi-GAN generator).
        n_speakers (int): n_speakers. Default: 0.
        gin_channels (int): dim of speaker embedding. Default: 0.
        use_sdp (bool): if ``True``, use StochasticDurationPredictor else DurationPredictor. Default: True.
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout_ratio: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[int],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_ratio = dropout_ratio
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        # c_text -> h_text, mu, sigma
        self.text_encoder = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, dropout_ratio
        )

        # linspec -> z (vae encoder using reparameterization trick)
        self.linspec_encoder = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )

        # z -> f_theta(z) (glow encoder using torch.flip for mixing data)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # z -> wav (vae decoder. Generator of HiFi-GAN)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, text, text_lengths, linspec, linspec_lengths, sid=None):
        # b: batch_size, t_s: text_len, t_t: spec_len
        text_encoded, mean_text, logs_text, text_mask = self.text_encoder(text, text_lengths)  # [b, d, t_s]
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, mean_linspec, logs_linspec, linspec_mask = self.linspec_encoder(
            linspec, linspec_lengths, g=g
        )  # [b, d, t_t]
        z_theta = self.flow(z, linspec_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_text)  # [b, d, t_s]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_text, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_theta**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_theta.transpose(1, 2), (mean_text * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (mean_text**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # [b, t_t, t_s]

            attn_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(linspec_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
            )  # duration [b, 1, t_t, t_s]

        w = attn.sum(2)  # [b, 1, t_s]
        if self.use_sdp:
            l_length = self.duration_predictor(text_encoded, text_mask, w, g=g)
            l_length = l_length / torch.sum(text_mask)
        else:
            logw_ = torch.log(w + 1e-6) * text_mask
            logw = self.duration_predictor(text_encoded, text_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(text_mask)  # for averaging

        # expand prior ([b, t_s, d] -> [b, t_t, d])
        mean_text = torch.matmul(attn.squeeze(1), mean_text.transpose(1, 2)).transpose(1, 2)  # [b, t_t, d]
        logs_text = torch.matmul(attn.squeeze(1), logs_text.transpose(1, 2)).transpose(1, 2)

        z_sliced, ids_slice = utils.rand_slice_3d_sequence(z, linspec_lengths, self.segment_size)
        wav_sliced_hat = self.dec(z_sliced, g=g)
        return (
            wav_sliced_hat,
            l_length,
            attn,
            ids_slice,
            text_mask,
            linspec_mask,
            (z, z_theta, mean_text, logs_text, mean_linspec, logs_linspec),
        )

    def infer(self, text, text_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1.0, max_len=None):
        text_encoded, mean_text, logs_text, text_mask = self.text_encoder(text, text_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.duration_predictor(text_encoded, text_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.duration_predictor(text_encoded, text_mask, g=g)
        w = torch.exp(logw) * text_mask * length_scale
        w_ceil = torch.ceil(w)
        linspec_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        linspec_mask = torch.unsqueeze(utils.make_mask(linspec_lengths, None), 1).to(text_mask.dtype)
        attn_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(linspec_mask, -1)
        attn = utils.generate_path(w_ceil, attn_mask)

        mean_text = torch.matmul(attn.squeeze(1), mean_text.transpose(1, 2)).transpose(1, 2)
        logs_text = torch.matmul(attn.squeeze(1), logs_text.transpose(1, 2)).transpose(1, 2)

        z_theta_hat = mean_text + torch.randn_like(mean_text) * torch.exp(logs_text) * noise_scale
        z_hat = self.flow(z_theta_hat, linspec_mask, g=g, reverse=True)
        wav_hat = self.dec((z_hat * linspec_mask)[:, :, :max_len], g=g)
        return wav_hat, attn, linspec_mask, (z_hat, z_theta_hat, mean_text, logs_text)

    def voice_conversion(self, linspec, linspec_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, mean_linspec, logs_linspec, linspec_mask = self.linspec_encoder(linspec, linspec_lengths, g=g_src)
        z_theta = self.flow(z, linspec_mask, g=g_src)
        z_hat = self.flow(z_theta, linspec_mask, g=g_tgt, reverse=True)
        wav_hat = self.dec(z_hat * linspec_mask, g=g_tgt)
        return wav_hat, linspec_mask, (z, z_theta, z_hat)
