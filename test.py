import torch
import random
import operator
import os
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from config import *
from model import Seq2Seq, Encoder, Decoder
from data_utils import get_dataloaders, tokenize_en

# --- 1. Hàm Beam Search (Phiên bản KHÔNG Attention) ---
def beam_search_decoding(sentence, model, vocab_en, vocab_fr, device, beam_width=3, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = sentence

    tokens = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(tokens)])

    with torch.no_grad():
        # Encoder: Chỉ lấy hidden, cell (context vector tĩnh)
        # Bỏ qua encoder_outputs vì không dùng Attention
        _, hidden, cell = model.encoder(src_tensor, src_len)

    sos_idx = vocab_fr['<sos>']
    eos_idx = vocab_fr['<eos>']
    
    beams = [(0.0, [sos_idx], hidden, cell)]
    
    for step in range(max_len):
        candidates = []
        
        for score, seq, h, c in beams:
            if seq[-1] == eos_idx:
                candidates.append((score, seq, h, c))
                continue
            
            dec_input = torch.LongTensor([seq[-1]]).to(device)
            
            with torch.no_grad():
                # DECODER MỚI: Chỉ truyền 3 tham số (input, hidden, cell)
                # KHÔNG truyền encoder_outputs và mask nữa
                output, new_h, new_c = model.decoder(dec_input, h, c)
                
                log_probs = F.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
            for i in range(beam_width):
                token_idx = top_indices[0][i].item()
                token_log_prob = top_log_probs[0][i].item()
                new_score = score + token_log_prob
                new_seq = seq + [token_idx]
                candidates.append((new_score, new_seq, new_h, new_c))
        
        ordered = sorted(candidates, key=operator.itemgetter(0), reverse=True)
        beams = ordered[:beam_width]
        
        if all(b[1][-1] == eos_idx for b in beams):
            break
            
    best_score, best_seq, _, _ = beams[0]
    trg_tokens = [vocab_fr.lookup_token(i) for i in best_seq]
    
    if '<sos>' in trg_tokens: trg_tokens.remove('<sos>')
    if '<eos>' in trg_tokens: trg_tokens.remove('<eos>')
            
    return trg_tokens

# --- 2. Tính BLEU Score ---
def calculate_bleu(data_loader, model, vocab_en, vocab_fr, device):
    trgs = []
    pred_trgs = []
    
    print(f"Đang tính BLEU score (No Attention)...")
    
    model.eval()
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(data_loader):
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            
            for j in range(src.shape[0]):
                src_indices = [idx.item() for idx in src[j] if idx not in [vocab_en['<pad>'], vocab_en['<eos>'], vocab_en['<sos>']]]
                src_tokens = [vocab_en.lookup_token(idx) for idx in src_indices]
                
                # Gọi hàm Beam Search đã sửa
                pred_tokens = beam_search_decoding(src_tokens, model, vocab_en, vocab_fr, device, beam_width=3)
                
                pred_trgs.append(pred_tokens)
                
                trg_indices = [idx.item() for idx in trg[j] if idx not in [vocab_fr['<pad>'], vocab_fr['<eos>'], vocab_fr['<sos>']]]
                trg_tokens = [vocab_fr.lookup_token(idx) for idx in trg_indices]
                trgs.append([trg_tokens])
            
            if (i + 1) % 10 == 0:
                print(f"Đã xử lý {i + 1} batch...", end='\r')

    print("\nHoàn tất dịch.")
    return corpus_bleu(trgs, pred_trgs)

if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_en, vocab_fr = get_dataloaders()
    
    enc = Encoder(len(vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint từ {MODEL_SAVE_PATH}...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        except RuntimeError as e:
            print("\nLỖI LOAD MODEL: File checkpoint không khớp với cấu trúc mô hình hiện tại.")
            print("Gợi ý: Bạn vừa tắt Attention? Hãy xóa file models/best_model.pth cũ và chạy train.py lại từ đầu.")
            exit()
    else:
        print(f"Không tìm thấy model tại {MODEL_SAVE_PATH}. Vui lòng chạy train.py trước!")
        exit()
    
    print("\n--- 5 Ví dụ Dịch (No Attention) ---")
    import random
    test_data_raw = test_loader.dataset
    indices = random.sample(range(len(test_data_raw)), 5)
    
    for i in indices:
        src_txt, trg_txt = test_data_raw[i]
        pred_tokens = beam_search_decoding(src_txt, model, vocab_en, vocab_fr, DEVICE, beam_width=3)
        pred_sent = " ".join(pred_tokens)
        
        print(f"Nguồn : {src_txt}")
        print(f"Đích  : {trg_txt}")
        print(f"Dự đoán: {pred_sent}")
        print("-" * 30)
        
    score = calculate_bleu(test_loader, model, vocab_en, vocab_fr, DEVICE)
    print(f'BLEU score trên tập Test: {score*100:.2f}')