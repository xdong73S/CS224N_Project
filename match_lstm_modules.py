import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax, masked_flip
import util

import math

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, hidden_size, char_drop_prob=0.05, word_drop_prob=0.1, 
                 use_char_emb=False):
        super(Embedding, self).__init__()
        self.use_char_emb = use_char_emb
        self.char_drop_prob = char_drop_prob
        self.word_drop_prob = word_drop_prob
        char_emb_dim = 64
        word_emb_dim = 300
        if use_char_emb:
            self.conv2d = nn.Conv2d(char_emb_dim, hidden_size, 
                                    kernel_size=(1, 5), padding=0, bias=True)
            nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
            proj_input_size = hidden_size + word_emb_dim
        else:
            proj_input_size = word_emb_dim

        self.proj = nn.Linear(proj_input_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_emb, char_emb=None):
        word_emb = F.dropout(word_emb, self.word_drop_prob, self.training)
        # (bs, seq_len, word_emb_dim)
        if self.use_char_emb:
            # (bs, seq_len, char_lim, char_emb_dim)
            char_emb = char_emb.permute(0, 3, 1, 2) # (bs, char_emb_dim, seq_len, char_lim)
            char_emb = F.dropout(char_emb, self.char_drop_prob, self.training)
            char_emb = self.conv2d(char_emb)
            char_emb = F.relu(char_emb)
            char_emb, _ = torch.max(char_emb, dim=3)
            char_emb = char_emb.squeeze()
            char_emb = char_emb.transpose(1, 2) # (bs, seq_len, hidden_size)
            emb = torch.cat([char_emb, word_emb], dim=-1)
        else:
            emb = word_emb
        
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class MatchRNNAttention(torch.nn.Module):
    r"""
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output
    Outputs:
        alpha(batch, question_len): attention vector
    """
    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()
        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (question_len, batch, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)  # (1, batch, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)  # (1, batch, hidden_size)
        G = F.tanh(wq_hq + wp_hp + wr_hr)  # (question_len, batch, hidden_size), auto broadcast
        wg_g = self.linear_wg(G) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, question_len)
        alpha = masked_softmax(wg_g, mask=Hq_mask, dim=1)  # (batch, question_len)
        return alpha

class UniMatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm
        rnn_in_size = hp_input_size + hq_input_size

        self.attention = MatchRNNAttention(hp_input_size, hq_input_size, hidden_size)

        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = Hq.new_zeros(batch_size, self.hidden_size)
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_para = {}
        vis_alpha = []
        vis_gated = []

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size)
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)  # (batch, question_len)
            vis_alpha.append(alpha)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, rnn_in_size)

            # gated
            if self.gated_attention:
                gate = F.sigmoid(self.gated_linear.forward(cur_z))
                vis_gated.append(gate.squeeze(-1))
                cur_z = gate * cur_z

            # layer normalization
            if self.enable_layer_norm:
                cur_z = self.layer_norm(cur_z)  # (batch, rnn_in_size)

            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])  # (batch, hidden_size), when lstm output tuple
            hidden.append(cur_hidden)

        vis_para['gated'] = torch.stack(vis_gated, dim=-1)  # (batch, context_len)
        vis_para['alpha'] = torch.stack(vis_alpha, dim=2)  # (batch, question_len, context_len)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)  # (context_len, batch, hidden_size)
        return result, vis_para

class MatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET
    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values
    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention,
                 dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2

        self.left_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                          enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                               enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)

        left_hidden, left_para = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden
        rtn_para = {'left': left_para}

        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden_inv, right_para_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)

            # flip back to normal sequence
            right_alpha_inv = right_para_inv['alpha']
            right_alpha_inv = right_alpha_inv.transpose(0, 1)  # make sure right flip
            right_alpha = masked_flip(right_alpha_inv, Hp_mask, flip_dim=2)
            right_alpha = right_alpha.transpose(0, 1)

            right_gated_inv = right_para_inv['gated']
            right_gated_inv = right_gated_inv.transpose(0, 1)
            right_gated = masked_flip(right_gated_inv, Hp_mask, flip_dim=2)
            right_gated = right_gated.transpose(0, 1)

            right_hidden = masked_flip(right_hidden_inv, Hp_mask, flip_dim=0)

            rtn_para['right'] = {'alpha': right_alpha, 'gated': right_gated}
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden, rtn_para

class PointerAttention(torch.nn.Module):
    r"""
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time
    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()

        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)  # (context_len, batch, hidden_size)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)  # (1, batch, hidden_size)
        f = F.tanh(wr_hr + wa_ha)  # (context_len, batch, hidden_size)

        beta_tmp = self.linear_wf(f) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, context_len)

        beta = masked_softmax(beta_tmp, mask=Hr_mask, dim=1)
        return beta

class UniBoundaryPointer(torch.nn.Module):
    r"""
    Unidirectional Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - enable_layer_norm: Whether use layer normalization
    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
        **hidden** (batch, hidden_size), [(batch, hidden_size)]: rnn last state
    """
    answer_len = 2

    def __init__(self, mode, input_size, hidden_size, enable_layer_norm):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enable_layer_norm = enable_layer_norm

        self.attention = PointerAttention(input_size, hidden_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = Hr.new_zeros(batch_size, self.hidden_size)

        hidden = (h_0, h_0) if self.mode == 'LSTM' and isinstance(h_0, torch.Tensor) else h_0
        beta_out = []

        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            if self.enable_layer_norm:
                context_beta = self.layer_norm(context_beta)  # (batch, input_size)

            hidden = self.hidden_cell.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)
        return result, hidden

class BoundaryPointer(torch.nn.Module):
    r"""
    Boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer
        - bidirectional: Bidirectional or Unidirectional
        - dropout_p: Dropout probability
        - enable_layer_norm: Whether use layer normalization
    Inputs:
        Hr (context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0 (batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size, enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        left_beta, _ = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0)
        rtn_beta = left_beta
        if self.bidirectional:
            right_beta_inv, _ = self.right_ptr_rnn.forward(Hr, Hr_mask, h_0)
            right_beta = right_beta_inv[[1, 0], :]

            rtn_beta = (left_beta + right_beta) / 2

        # todo: unexplainable
        Hr_mask = Hr_mask.type(torch.float32)
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)
        #rtn_beta = torch.log(rtn_beta)

        return rtn_beta

