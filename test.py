import torch
import random
import operator
import os 
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from config import *
from model import Seq2Seq, Encoder, Decoder
from data_utils import get_dataloaders, tokenize_en

# --- 1. Hàm Beam Search Decoding ---
def beam_search_decoding(sentence, model, vocab_en, vocab_fr, device, beam_width=5, max_len=50):
    """
    Thực hiện giải mã Beam Search để tìm câu dịch tốt nhất.
    """
    model.eval()

    # 1. Xử lý Input (Encoder)
    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = sentence

    # Chuyển tokens thành tensor
    tokens = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device) # [src_len, 1]
    src_len = torch.LongTensor([len(tokens)])

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

    # Tạo mask cho encoder outputs
    mask = torch.zeros(1, src_tensor.size(0), dtype=torch.bool, device=device)
    mask[0, :src_len.item()] = 1

    # 2. Khởi tạo Beam: (score, sequence, hidden, cell)
    sos_idx = vocab_fr['<sos>']
    eos_idx = vocab_fr['<eos>']
    
    beams = [(0.0, [sos_idx], hidden, cell)]
    
    # 3. Vòng lặp Decoding
    for step in range(max_len):
        candidates = []
        
        for score, seq, h, c in beams:
            # Nếu beam này đã kết thúc bằng <eos>, giữ nguyên
            if seq[-1] == eos_idx:
                candidates.append((score, seq, h, c))
                continue
            
            dec_input = torch.LongTensor([seq[-1]]).to(device)
            
            with torch.no_grad():
                output, new_h, new_c = model.decoder(dec_input, h, c, encoder_outputs, mask)
                log_probs = F.log_softmax(output, dim=1)
                
                # Lấy Top K ứng cử viên
                top_log_probs, top_indices = log_probs.topk(beam_width)
                
            for i in range(beam_width):
                token_idx = top_indices[0][i].item()
                token_log_prob = top_log_probs[0][i].item()
                new_score = score + token_log_prob
                new_seq = seq + [token_idx]
                candidates.append((new_score, new_seq, new_h, new_c))
        
        # Sắp xếp và chọn ra beam_width nhánh tốt nhất
        ordered = sorted(candidates, key=operator.itemgetter(0), reverse=True)
        beams = ordered[:beam_width]
        
        # Dừng nếu tất cả các nhánh đều đã gặp <eos>
        if all(b[1][-1] == eos_idx for b in beams):
            break
            
    # 4. Lấy kết quả tốt nhất (Beam đầu tiên)
    best_score, best_seq, _, _ = beams[0]
    trg_tokens = [vocab_fr.lookup_token(i) for i in best_seq]
    
    # Loại bỏ token đặc biệt
    if '<sos>' in trg_tokens: trg_tokens.remove('<sos>')
    if '<eos>' in trg_tokens: trg_tokens.remove('<eos>')
            
    return trg_tokens

# --- 2. Tính BLEU Score ---
def calculate_bleu(data_loader, model, vocab_en, vocab_fr, device):
    trgs = []
    pred_trgs = []
    
    print(f"Đang tính BLEU score bằng BEAM SEARCH ...")
    
    model.eval()
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(data_loader):
            src = src.transpose(0, 1) # [batch, src_len]
            trg = trg.transpose(0, 1) # [batch, trg_len]
            
            for j in range(src.shape[0]):
                # Lấy câu gốc
                src_indices = [idx.item() for idx in src[j] if idx not in [vocab_en['<pad>'], vocab_en['<eos>'], vocab_en['<sos>']]]
                src_tokens = [vocab_en.lookup_token(idx) for idx in src_indices]
                
                # Dịch bằng Beam Search (Trực tiếp)
                pred_tokens = beam_search_decoding(src_tokens, model, vocab_en, vocab_fr, device, beam_width=3)
                
                pred_trgs.append(pred_tokens)
                
                # Lấy câu đích (Ground Truth)
                trg_indices = [idx.item() for idx in trg[j] if idx not in [vocab_fr['<pad>'], vocab_fr['<eos>'], vocab_fr['<sos>']]]
                trg_tokens = [vocab_fr.lookup_token(idx) for idx in trg_indices]
                trgs.append([trg_tokens])
            
            #tiến độ
            if (i + 1) % 10 == 0:
                print(f"Đã xử lý {i + 1} batch...", end='\r')

    print("\nHoàn tất dịch.")
    return corpus_bleu(trgs, pred_trgs)

if __name__ == "__main__":
    # --- LOAD ---
    train_loader, val_loader, test_loader, vocab_en, vocab_fr = get_dataloaders()
    
    # Khởi tạo mô hình
    enc = Encoder(len(vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    # Kiểm tra và load checkpoint
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint từ {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print(f"LỖI: Không tìm thấy model tại {MODEL_SAVE_PATH}. Vui lòng chạy train.py trước!")
        exit()
    
    # --- 1. DỊCH THỬ 5 VÍ DỤ ---
    print("\n--- 5 Ví dụ Dịch (Sử dụng Beam Search) ---")
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
        
    # --- 2. TÍNH BLEU TOÀN BỘ TẬP TEST ---
    score = calculate_bleu(test_loader, model, vocab_en, vocab_fr, DEVICE)
    print(f'BLEU score trên tập Test: {score*100:.2f}')