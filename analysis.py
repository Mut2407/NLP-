# analysis.py (Mới hoàn toàn - Dùng để viết báo cáo Phần 9)
import torch
from config import *
from data_utils import get_dataloaders
from model import Encoder, Decoder, Seq2Seq
from nltk.translate.bleu_score import sentence_bleu

def load_model(vocab_en, vocab_fr):
    enc = Encoder(len(vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    return model

def translate(sentence_tensor, model, vocab_fr, max_len=50):
    with torch.no_grad():
        src_len = torch.LongTensor([len(sentence_tensor)])
        hidden, cell = model.encoder(sentence_tensor.unsqueeze(1).to(DEVICE), src_len)
        
        trg_indexes = [vocab_fr['<sos>']]
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == vocab_fr['<eos>']: break
            
    return [vocab_fr.lookup_token(i) for i in trg_indexes[1:-1]]

def analyze_errors():
    _, _, test_loader, vocab_en, vocab_fr = get_dataloaders()
    model = load_model(vocab_en, vocab_fr)
    
    print("\n=== PHẦN 9: PHÂN TÍCH LỖI DỊCH THUẬT ===")
    
    unk_errors = []
    long_sentence_errors = []
    
    # Duyệt qua tập test (Lấy 1 batch đại diện hoặc loop hết nếu cần)
    for src_batch, trg_batch, src_lens in test_loader:
        src_batch = src_batch.transpose(0, 1) # [batch, len]
        trg_batch = trg_batch.transpose(0, 1)
        
        for i in range(min(len(src_batch), 100)): # Kiểm tra 100 câu đầu
            # 1. Lấy câu gốc và câu đích
            src_indices = [x.item() for x in src_batch[i] if x not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]]
            trg_indices = [x.item() for x in trg_batch[i] if x not in [vocab_fr['<pad>'], vocab_fr['<sos>'], vocab_fr['<eos>']]]
            
            src_text = [vocab_en.lookup_token(x) for x in src_indices]
            trg_text = [vocab_fr.lookup_token(x) for x in trg_indices]
            
            # 2. Dự đoán
            src_tensor = torch.LongTensor([vocab_en['<sos>']] + src_indices + [vocab_en['<eos>']])
            pred_text = translate(src_tensor, model, vocab_fr)
            
            # 3. Phân tích lỗi 1: Từ hiếm (OOV -> <unk>)
            if '<unk>' in pred_text:
                unk_errors.append((src_text, pred_text, trg_text))
                
            # 4. Phân tích lỗi 2: Câu dài (>15 từ) thường bị mất thông tin
            if len(src_text) > 15:
                bleu = sentence_bleu([trg_text], pred_text)
                if bleu < 0.3: # Điểm thấp
                    long_sentence_errors.append((src_text, pred_text, bleu))

        break # Chỉ chạy 1 batch để demo nhanh

    print(f"\n[Lỗi 1] Từ hiếm (OOV - <unk>) - Tìm thấy {len(unk_errors)} câu:")
    for src, pred, trg in unk_errors[:3]:
        print(f"SRC : {' '.join(src)}")
        print(f"PRED: {' '.join(pred)}")
        print(f"TRG : {' '.join(trg)}")
        print("-" * 20)

    print(f"\n[Lỗi 2] Câu dài mất thông tin (Low BLEU) - Tìm thấy {len(long_sentence_errors)} câu:")
    for src, pred, bleu in long_sentence_errors[:3]:
        print(f"SRC ({len(src)} words): {' '.join(src)}")
        print(f"PRED: {' '.join(pred)}")
        print(f"BLEU: {bleu:.4f}")
        print("-" * 20)
        
    # print("\n=> Đề xuất cải tiến (Viết vào báo cáo):")
    # print("1. Thêm cơ chế Attention (Bahdanau/Luong) để model tập trung vào từng từ thay vì nén hết vào Context Vector.")
    # print("2. Sử dụng Subword Tokenization (BPE) để xử lý từ hiếm <unk>.")
    # print("3. Thay Greedy Decoding bằng Beam Search để tìm câu tốt hơn.")

if __name__ == "__main__":
    analyze_errors()