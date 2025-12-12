import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
from config import *
from data_utils import get_dataloaders
from model import Encoder, Decoder, Seq2Seq

# --- CLASS EARLY STOPPING (Yêu cầu Phần 8 - Điểm 1.5) ---
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        Dừng train nếu validation loss không giảm sau 'patience' epochs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'INFO: Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg, src_len) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg, src_len)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, src_len, 0) # Tắt teacher forcing
            
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

if __name__ == "__main__":
    # 1. Load Data
    train_loader, val_loader, _, vocab_en, vocab_fr = get_dataloaders()
    
    INPUT_DIM = len(vocab_en)
    OUTPUT_DIM = len(vocab_fr)
    TRG_PAD_IDX = vocab_fr['<pad>']

    # 2. Init Model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
    
    # Kích hoạt Early Stopping
    early_stopper = EarlyStopping(patience=3)

    # List để lưu lịch sử Loss phục vụ vẽ biểu đồ
    train_losses = []
    valid_losses = []

    print(f"Bắt đầu huấn luyện trên {DEVICE}...")
    print(f"Early Stopping patience: 3 epochs")
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_loader, criterion)
        
        # Lưu loss vào list
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        #Tính ppl
        train_ppls = [math.exp(l) for l in train_losses]
        valid_ppls = [math.exp(l) for l in valid_losses]
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Check Early Stopping & Save Best Model
        if valid_loss < early_stopper.best_loss:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\t--> Đã lưu model tốt nhất (Val Loss giảm còn {valid_loss:.3f})")
            
        early_stopper(valid_loss)
        
        print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')
        print(f'\tTrain PPL: {math.exp(train_loss):.3f} | Val PPL: {math.exp(valid_loss):.3f}')
        
        if early_stopper.early_stop:
            print("Dừng huấn luyện sớm do Loss không giảm (Early Stopping)!")
            break

    # --- VẼ BIỂU ĐỒ (Yêu cầu Tiêu chí số 5 - Điểm 1.0) ---
    print("\nBiểu đồ Loss...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(train_losses) + 1))
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/loss_chart.png') 
    print("File biểu đồ là 'loss_chart.png'")
    
    print("\nBiểu đồ PPL...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_ppls, label='Train PPL', marker='o')
    plt.plot(valid_ppls, label='Validation PPL', marker='x')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(train_ppls) + 1))
    plt.ylabel('PPL')
    plt.title('Training and Validation Perplexity Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('charts/ppl_chart.png')
    print("File biểu đồ là 'ppl_chart.png'")
    # plt.show() # Bỏ comment dòng này nếu muốn hiện cửa sổ ảnh (có thể lỗi trên một số server)