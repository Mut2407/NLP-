import torch
import os
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from config import *
from model import Seq2Seq, Encoder, Decoder
from data_utils import get_dataloaders, tokenize_en


# --- 1. Greedy Decoding (NO ATTENTION) ---
def greedy_decoding(sentence, model, vocab_en, vocab_fr, device, max_len=50):
    model.eval()

    # Tokenize câu nguồn
    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = sentence

    tokens = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(tokens)])

    with torch.no_grad():
        # Encoder: chỉ lấy hidden, cell 
        _, hidden, cell = model.encoder(src_tensor, src_len)

    sos_idx = vocab_fr['<sos>']
    eos_idx = vocab_fr['<eos>']

    trg_indexes = [sos_idx]
    input_token = torch.LongTensor([sos_idx]).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            # Decoder KHÔNG Attention → chỉ truyền input, hidden, cell
            output, hidden, cell = model.decoder(input_token, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == eos_idx:
            break

        input_token = torch.LongTensor([pred_token]).to(device)

    trg_tokens = [vocab_fr.lookup_token(i) for i in trg_indexes]
    trg_tokens = [t for t in trg_tokens if t not in ['<sos>', '<eos>', '<pad>']]

    return trg_tokens


# --- 2. Tính BLEU Score ---
def calculate_bleu(data_loader, model, vocab_en, vocab_fr, device):
    trgs = []
    pred_trgs = []

    print("Đang tính BLEU score (Greedy-No Attention)...")

    model.eval()
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(data_loader):
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)

            for j in range(src.shape[0]):
                src_indices = [
                    idx.item() for idx in src[j]
                    if idx not in [vocab_en['<pad>'], vocab_en['<sos>'], vocab_en['<eos>']]
                ]
                src_tokens = [vocab_en.lookup_token(idx) for idx in src_indices]

                pred_tokens = greedy_decoding(
                    src_tokens, model, vocab_en, vocab_fr, device
                )
                pred_trgs.append(pred_tokens)

                trg_indices = [
                    idx.item() for idx in trg[j]
                    if idx not in [vocab_fr['<pad>'], vocab_fr['<sos>'], vocab_fr['<eos>']]
                ]
                trg_tokens = [vocab_fr.lookup_token(idx) for idx in trg_indices]
                trgs.append([trg_tokens])

            if (i + 1) % 10 == 0:
                print(f"Đã xử lý {i + 1} batch...", end="\r")

    print("\nHoàn tất dịch.")
    return corpus_bleu(trgs, pred_trgs)


# --- MAIN ---
if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_en, vocab_fr = get_dataloaders()

    enc = Encoder(len(vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint từ {MODEL_SAVE_PATH}...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("Không tìm thấy model. Hãy chạy train.py trước!")
        exit()

    # --- DỊCH THỬ ---
    print("\n--- 5 Ví dụ Dịch (Greedy – No Attention) ---")
    test_data_raw = test_loader.dataset
    indices = torch.randperm(len(test_data_raw))[:5]

    for i in indices:
        src_txt, trg_txt = test_data_raw[i]
        pred_tokens = greedy_decoding(src_txt, model, vocab_en, vocab_fr, DEVICE)
        pred_sent = " ".join(pred_tokens)

        print(f"Nguồn   : {src_txt}")
        print(f"Đích    : {trg_txt}")
        print(f"Dự đoán : {pred_sent}")
        print("-" * 40)

    score = calculate_bleu(test_loader, model, vocab_en, vocab_fr, DEVICE)
    print(f"BLEU score (Greedy – No Attention): {score * 100:.2f}")
