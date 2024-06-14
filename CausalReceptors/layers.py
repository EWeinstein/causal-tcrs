"""
Code organization:
NN layers:
    - SeqEmbed: layer for mapping individual sequences to sequence representations.
    - SeqEmbedToRepertoireEmbed: layer for mapping sequence representations to repertoire representations.
    - RepertoireEmbedToSelectionEmbed: layer for mapping repertoire representations to selection representations.
    - SelectionEmbedToRepertoireEmbed: layer for mapping selection representations to repertoire representations.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn


def one_hot(data, num_letter, dtype=torch.float32):
    """One-hot encode sequences. -1 is encoded to all zeros (missing data)"""
    return torch.nn.functional.one_hot(data + 1, num_letter + 1)[..., 1:].to(dtype)


class KmerEmbed(nn.Module):
    def __init__(self, num_length, alphabet, kmer_length, custom_kmers=None, dtype=torch.float32, cuda=False):
        """Initialization"""
        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.dtype = dtype

        # Focus on kmers w/o stop codons.
        self.alphabet_length = len(alphabet)
        if alphabet[-1] == '*':
            self.alphabet_length -= 1

        if custom_kmers is None:
            # Count each possible kmer (w/o the stop symbol) of length kmer_length.
            self.n_kmers = self.alphabet_length ** kmer_length
            kmers = torch.zeros(list(map(int, self.alphabet_length * np.ones(kmer_length))) + [kmer_length],
                                dtype=torch.long, device=self.device)
            for j in range(kmer_length):
                x = torch.arange(self.alphabet_length, device=self.device)
                for jp in range(kmer_length):
                    if j != jp:
                        x = x.unsqueeze(jp)
                kmers[..., j] += x
            self.kmers = kmers.reshape([-1, kmer_length])

        else:
            # Load custom kmers.
            self.kmers = custom_kmers.to(dtype=torch.long, device=self.device)

        self.n_kmers = self.kmers.shape[0]
        self.weight = torch.nn.functional.one_hot(self.kmers, self.alphabet_length).swapaxes(-2, -1).to(dtype=dtype)
        self.bias = (1 - kmer_length) * torch.ones(self.n_kmers, dtype=dtype, device=self.device)

    def forward(self, seqs):
        # In: N x M x L x num_letter
        N, M, L, B = seqs.shape
        # Reshape. Out: (N x M) x L x alphabet_length.
        oh = seqs[:, :, :, :self.alphabet_length].view([N * M, L, self.alphabet_length])
        # Convolve, threshold and sum over positions.
        kmer_embed = torch.nn.functional.conv1d(oh.swapaxes(1, 2), self.weight, bias=self.bias).relu().sum(dim=-1)
        # Reshape and return.
        return kmer_embed.view((N, M, self.n_kmers))


class SeqEmbed(nn.Module):
    """Neural network for mapping sequences to sequence representations"""

    def __init__(self, num_length, alphabet, channels, conv_kernel=5,
                 architecture='cnn', pos_encode=False, sum_pool=False, linear_cnn=False, no_pool=False,
                 transformer_nhead=8, transformer_dimff=64, dtype=torch.float32, cuda=False):
        """Initialization."""
        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.dtype = dtype

        # Position encoding.
        self.input_feats = len(alphabet)
        self.pos_encode = pos_encode
        if self.pos_encode:
            # Using the position encoding of Widrich et al.
            self.input_feats += 3

        # Sequence representation.
        self.architecture = architecture
        if architecture == 'cnn':
            # Based on DeepRC CNN architecture.
            cnn = []
            # Convolution.
            self.num_letter = len(alphabet)
            conv = nn.Conv1d(self.input_feats, channels, conv_kernel, dtype=dtype)
            conv.weight.data.normal_(0.0, np.sqrt(1 / np.prod(conv.weight.shape)))
            cnn.append(conv)
            self.conv_channels = channels
            # Nonlinearity.
            if not linear_cnn:
                cnn.append(nn.SELU())
            self.hidden_size = num_length - conv_kernel + 1
            self.hidden_dim = channels
            self.cnn = nn.Sequential(*cnn)
            if cuda:
                self.cnn.cuda()
            self.to(dtype=dtype)

        elif architecture == 'transformer':
            self.transformer = nn.TransformerEncoderLayer(
                self.input_feats, transformer_nhead, dim_feedforward=transformer_dimff,
                batch_first=True, dropout=0.)
            self.hidden_size = num_length
            self.hidden_dim = self.input_feats
            if cuda:
                self.transformer.cuda()

        elif architecture == 'gru':
            self.gru = nn.GRU(input_size=self.input_feats, hidden_size=channels, batch_first=True)
            self.hidden_size = num_length
            self.hidden_dim = channels
            if cuda:
                self.gru.cuda()

        # Sum instead of taking maximum over positions.
        self.sum_pool = sum_pool

        # Return all channels at all positions, instead of pooling.
        self.no_pool = no_pool

    def forward(self, seqs):

        # In: N x M x L x input_feats
        N, M, L, _ = seqs.shape
        # Reshape. Out: (N x M) x L x num_letter.
        oh = seqs.view([N * M, L, self.input_feats])
        if self.architecture == 'cnn':
            # Swap axes. Out: (N x M) x input_feats x L.
            # 1D Convolutional NN. Out: (N x M) x conv_channels (hidden_dim) x hidden_size
            hidden = self.cnn(oh.swapaxes(1, 2))
        elif self.architecture == 'transformer':
            # Transformer. Out: (N x M) x L (hidden_size) x input_feats (hidden_dim)
            # Swap axes. Out: (N x M) x input_feats (hidden_dim) x L (hidden_size)
            hidden = self.transformer(oh).swapaxes(1, 2)
        elif self.architecture == 'gru':
            # GRU. Out: (N x M) x L (hidden_size) x conv_channels (hidden_dim)
            # Swap axes. Out: (N x M) x conv_channels (hidden_dim) x L (hidden_size)
            hidden = self.gru(oh)[0].swapaxes(1, 2)

        if self.no_pool:
            return hidden.view((N, M, self.hidden_size * self.hidden_dim))

        if self.sum_pool:
            # Mean over sequence positions. Out: (N x M) x conv_channels (hidden_dim)
            embed = hidden.sum(dim=-1)
        else:
            # Maximum over sequence positions. Out: (N x M) x conv_channels (hidden_dim)
            embed = hidden.max(dim=-1)[0]

        return embed.view((N, M, self.hidden_dim))


class AttentionReduce(nn.Module):
    """Reduce (featurized) repertoire sequences to single vector per patient."""

    def __init__(self, n_input_features, n_attention_layers, n_attention_units,
                 no_attention=False, no_embedding=False, top_fraction=1., use_counts=True, cuda=False):
        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_input_features = n_input_features
        self.n_attention_layers = n_attention_layers
        self.n_attention_units = n_attention_units
        self.top_fraction = top_fraction
        self.no_attention = no_attention
        self.no_embedding = no_embedding  # Deprecated.
        # Weight by number of counts of each sequence.
        self.use_counts = use_counts

        if not no_attention:
            # Attention mechanism (following Widrich et al. 2020).
            fc_attention = []
            for _ in range(n_attention_layers):
                att_linear = nn.Linear(n_input_features, n_attention_units)
                att_linear.weight.data.normal_(0., np.sqrt(1 / np.prod(att_linear.weight.shape)))
                fc_attention.append(att_linear)
                fc_attention.append(nn.SELU())
                n_input_features = n_attention_units

            att_linear = nn.Linear(n_input_features, 1)
            att_linear.weight.data.normal_(0.0, np.sqrt(1 / np.prod(att_linear.weight.shape)))
            fc_attention.append(att_linear)
            self.attn = torch.nn.Sequential(*fc_attention)
            if cuda:
                self.attn.cuda()

            # Take sequences with top attention values.
            if np.allclose(top_fraction, 1.):
                self.take_top = False
            else:
                self.take_top = True

    def forward(self, embed, counts=None):
        # Input: Patients by sequences by sequence features
        N, M, ed = embed.shape

        if self.no_attention:
            # Average sequence embeddings to obtain repertoire embedding.
            if self.use_counts:
                return (embed * counts[:, :, None] / counts.sum(dim=1)[:, None, None]).sum(dim=1), torch.ones(N)
            else:
                return embed.mean(dim=1), torch.ones(N)

        # Attention scores.
        atts = self.attn(embed.view((N * M, ed))).view([N, M])

        if self.use_counts:
            counts_ln = counts.log()
            atts += counts_ln - counts_ln.logsumexp(dim=1, keepdim=True)

        if self.no_embedding:
            # Deprecated.
            if self.use_counts:
                assert False, 'Not yet implemented'
            else:
                return atts.mean(dim=1, keepdim=True)

        # Take top k
        if self.take_top:
            topk = int(N * self.top_fraction)
            atts, topindx = torch.topk(atts, topk, dim=1, largest=True, sorted=False)
            embed = torch.gather(embed, 1, torch.tile(topindx[:, :, None], (1, 1, ed)))

        # Attention weights. attn: N x 1, attw: N x M
        # Weighted average. enc_hidden: N x embed
        # Note: this is equivalent to atts.softmax, but we record the denominator to compute effects efficiently
        attn = atts.logsumexp(dim=1, keepdim=True)
        attw = (atts - attn).exp()
        enc_hidden = (embed * attw[:, :, None]).sum(dim=1)

        return enc_hidden, attn.squeeze(dim=1)