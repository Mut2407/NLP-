import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --- ENCODER (Giữ nguyên) ---
# Dù không dùng outputs cho attention, ta vẫn giữ nguyên cấu trúc
# để đảm bảo hidden/cell được tính toán đúng qua LSTM
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))

        # Packing sequence 
        packed_embedded = pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs)
        
        # Trả về hidden, cell làm context vector khởi tạo cho Decoder
        return outputs, hidden, cell

# --- DECODER (Đã tháo Attention) ---
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        # THAY ĐỔI: Input của Linear chỉ là hid_dim (không còn ghép với context vector)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch size]
        input = input.unsqueeze(0) # [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # Decoder chạy từng bước thời gian
        # Input: embedded, hidden t-1, cell t-1
        # Output: output, hidden t, cell t
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # THAY ĐỔI: output shape = [1, batch size, hid_dim]
        # Bỏ qua bước tính attention và concat
        prediction = self.fc_out(output.squeeze(0)) 
        
        return prediction, hidden, cell

# --- SEQ2SEQ (Kết hợp Encoder-Decoder Không Attention) ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 1. Encoder: Chỉ lấy hidden và cell để làm Context Vector ban đầu
        # encoder_outputs bị bỏ qua vì không có Attention để dùng nó
        _, hidden, cell = self.encoder(src, src_len)
        
        input = trg[0, :]

        for t in range(1, trg_len):
            # 2. Decoder: Không truyền encoder_outputs và mask nữa
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            # Teacher Forcing
            best_guess = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs