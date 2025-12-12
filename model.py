# model.py
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence

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
        
        # hidden, cell: Context Vector được truyền sang Decoder
        return hidden, cell

# --- DECODER (Hình 6.2) ---
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch_size] (1 token tại thời điểm t)
        input = input.unsqueeze(0) # [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        
        # Decoder chạy từng bước thời gian
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        prediction = self.fc_out(output.squeeze(0)) # [batch_size, output_dim]
        return prediction, hidden, cell

# --- SEQ2SEQ (Kết hợp Encoder-Decoder) ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Lấy Context Vector từ Encoder
        hidden, cell = self.encoder(src, src_len)
        
        # Token đầu tiên vào Decoder là <sos>
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            
            # Teacher Forcing: Quyết định dùng Ground Truth hay Prediction cho bước tiếp theo
            best_guess = output.argmax(1) 
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess
            
        return outputs