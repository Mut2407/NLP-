import torch
import random
from nltk.translate.bleu_score import corpus_bleu
from config import *
from model import Seq2Seq, Encoder, Decoder
from data_utils import get_dataloaders, tokenize_en

# 1. Hàm translate (Dùng Greedy Decoding)
def translate_sentence(sentence, model, vocab_en, vocab_fr, device, max_len=50):
    model.eval()
    
    # Xử lý input text
    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = sentence
    
    tokens = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(tokens)])
    
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
        
    trg_indexes = [vocab_fr['<sos>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vocab_fr['<eos>']:
            break
            
    trg_tokens = [vocab_fr.lookup_token(i) for i in trg_indexes]
    return trg_tokens[1:-1] # Bỏ sos, eos

# 2. Tính BLEU Score
def calculate_bleu(data_loader, model, vocab_en, vocab_fr, device):
    trgs = []
    pred_trgs = []
    
    print("Đang tính BLEU score (có thể mất vài phút)...")
    model.eval()
    with torch.no_grad():
        for src, trg, src_len in data_loader:
            src = src.transpose(0, 1) # [batch, src_len]
            trg = trg.transpose(0, 1) # [batch, trg_len]
            
            for i in range(src.shape[0]):
                # Lấy câu gốc (loại bỏ pad, eos)
                src_indices = [idx.item() for idx in src[i] if idx not in [vocab_en['<pad>'], vocab_en['<eos>'], vocab_en['<sos>']]]
                src_tokens = [vocab_en.lookup_token(idx) for idx in src_indices]
                
                # Dịch
                pred_tokens = translate_sentence(src_tokens, model, vocab_en, vocab_fr, device)
                pred_trgs.append(pred_tokens)
                
                # Lấy câu đích (Ground Truth)
                trg_indices = [idx.item() for idx in trg[i] if idx not in [vocab_fr['<pad>'], vocab_fr['<eos>'], vocab_fr['<sos>']]]
                trg_tokens = [vocab_fr.lookup_token(idx) for idx in trg_indices]
                trgs.append([trg_tokens])
                
    return corpus_bleu(trgs, pred_trgs)

if __name__ == "__main__":
    # --- LOAD ---
    train_loader, val_loader, test_loader, vocab_en, vocab_fr = get_dataloaders()
    
    enc = Encoder(len(vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    print(f"Loading checkpoint từ {MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    
    # --- TEST ---
    print("\n--- 5 Ví dụ Dịch ---")
    # Lấy ngẫu nhiên từ test set
    import random
    test_data_raw = test_loader.dataset
    indices = random.sample(range(len(test_data_raw)), 5)
    
    for i in indices:
        src_txt, trg_txt = test_data_raw[i]
        pred_tokens = translate_sentence(src_txt, model, vocab_en, vocab_fr, DEVICE)
        pred_sent = " ".join(pred_tokens)
        
        print(f"SRC : {src_txt}")
        print(f"TRG : {trg_txt}")
        print(f"PRED: {pred_sent}")
        print("-" * 30)
        
    # --- TÍNH BLEU ---
    # Chạy trên tập test
    score = calculate_bleu(test_loader, model, vocab_en, vocab_fr, DEVICE)
    print(f'BLEU score trên tập Test: {score*100:.2f}')