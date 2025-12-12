# model.py
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --- ENCODER (Hình 6.2) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # Packing sequence (Yêu cầu bắt buộc để tối ưu LSTM với padding)
        packed_embedded = pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)

        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        # Unpack to get encoder outputs for attention
        outputs, _ = pad_packed_sequence(packed_outputs)

        # outputs: [src_len, batch_size, hid_dim]
        # hidden, cell: Context Vector được truyền sang Decoder
        return outputs, hidden, cell

# --- DECODER (Hình 6.2) ---
class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        # Optional projection for general score function
        self.attn = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [n_layers, batch, hid_dim] -> use last layer
        # encoder_outputs: [src_len, batch, hid_dim]
        hidden_last = hidden[-1]  # [batch, hid_dim]
        # Apply linear
        proj_hidden = self.attn(hidden_last)  # [batch, hid_dim]
        # Compute energies: batch x src_len
        enc = encoder_outputs.transpose(0, 1)  # [batch, src_len, hid_dim]
        energies = torch.bmm(enc, proj_hidden.unsqueeze(2)).squeeze(2)  # [batch, src_len]
        if mask is not None:
            energies = energies.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(energies, dim=1)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), enc).squeeze(1)  # [batch, hid_dim]
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.attention = LuongAttention(hid_dim)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, src_mask=None):
        # input: [batch_size] (1 token tại thời điểm t)
        input = input.unsqueeze(0)  # [1, batch_size]

        embedded = self.dropout(self.embedding(input))

        # Decoder chạy từng bước thời gian
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        dec_out = output.squeeze(0)  # [batch, hid_dim]

        # Attention over encoder outputs
        context, _ = self.attention(hidden, encoder_outputs, src_mask)
        combined = torch.cat([dec_out, context], dim=1)  # [batch, hid*2]

        prediction = self.fc_out(combined)  # [batch, output_dim]
        return prediction, hidden, cell

# --- SEQ2SEQ (Kết hợp Encoder-Decoder) ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def make_src_mask(self, src, src_len):
        # Create mask with 1 for valid tokens and 0 for padding
        # src: [src_len, batch]
        max_len = src.size(0)
        batch_size = src.size(1)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            valid_len = src_len[i].item()
            mask[i, :valid_len] = 1
        return mask  # [batch, src_len]

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder outputs + hidden state
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        src_mask = self.make_src_mask(src, src_len)

        # Token đầu tiên vào Decoder là <sos>
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, src_mask)
            outputs[t] = output

            # Teacher Forcing: Quyết định dùng Ground Truth hay Prediction cho bước tiếp theo
            best_guess = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs