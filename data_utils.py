# data_utils.py (Phiên bản KHÔNG DÙNG torchtext)
import torch
import spacy
import io
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import *

# Load Spacy Tokenizers
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')
except OSError:
    print("Đang tải model ngôn ngữ cho Spacy...")
    from spacy.cli import download
    download("en_core_web_sm")
    download("fr_core_news_sm")
    spacy_en = spacy.load('en_core_web_sm')
    spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

# Class Vocab tùy chỉnh để thay thế torchtext.vocab
class Vocab:
    def __init__(self, frequency_dict, specials, max_tokens):
        self.itos = list(specials) # Index to String
        # Thêm các từ phổ biến nhất vào danh sách
        for word, _ in frequency_dict.most_common(max_tokens - len(specials)):
            self.itos.append(word)
            
        self.stoi = {word: i for i, word in enumerate(self.itos)} # String to Index
        self.default_index = self.stoi.get('<unk>')
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
        
    def __len__(self):
        return len(self.itos)
        
    def lookup_token(self, index):
        return self.itos[index]
        
    def set_default_index(self, index):
        self.default_index = index

# 1. Dataset Class
class EnFrDataset(Dataset):
    def __init__(self, en_path, fr_path):
        self.en_sentences = self.read_file(en_path)
        self.fr_sentences = self.read_file(fr_path)
        assert len(self.en_sentences) == len(self.fr_sentences), \
            f"Lỗi: Số dòng file {en_path} và {fr_path} không bằng nhau!"

    def read_file(self, filepath):
        with io.open(filepath, encoding="utf8") as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        return self.en_sentences[idx], self.fr_sentences[idx]

# 2. Xây dựng Vocabulary (Viết lại không dùng torchtext)
def build_vocabularies(train_dataset):
    print("Đang xây dựng từ điển (Vocabulary)...")
    
    counter_en = Counter()
    counter_fr = Counter()
    
    # Duyệt qua data để đếm từ
    for en, fr in train_dataset:
        counter_en.update(tokenize_en(en))
        counter_fr.update(tokenize_fr(fr))
        
    specials = ['<unk>', '<pad>', '<sos>', '<eos>']
    
    # Tạo object Vocab
    vocab_en = Vocab(counter_en, specials=specials, max_tokens=INPUT_DIM)
    vocab_fr = Vocab(counter_fr, specials=specials, max_tokens=OUTPUT_DIM)
    
    return vocab_en, vocab_fr

# 3. Collate Function
def create_collate_fn(vocab_en, vocab_fr):
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        src_lens = []
        
        for src_sample, trg_sample in batch:
            src_tokens = [vocab_en['<sos>']] + [vocab_en[token] for token in tokenize_en(src_sample)] + [vocab_en['<eos>']]
            trg_tokens = [vocab_fr['<sos>']] + [vocab_fr[token] for token in tokenize_fr(trg_sample)] + [vocab_fr['<eos>']]
            
            src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
            trg_batch.append(torch.tensor(trg_tokens, dtype=torch.long))
            src_lens.append(len(src_tokens))
            
        src_batch = pad_sequence(src_batch, padding_value=vocab_en['<pad>'])
        trg_batch = pad_sequence(trg_batch, padding_value=vocab_fr['<pad>'])
        
        return src_batch, trg_batch, torch.tensor(src_lens)
    return collate_fn

# Hàm chính
def get_dataloaders():
    import os
    # Tạo dummy data nếu chưa có file
    if not os.path.exists(TRAIN_EN_PATH):
        print("CẢNH BÁO: Không tìm thấy file dữ liệu. Đang tạo dữ liệu giả...")
        os.makedirs('data', exist_ok=True)
        dummy_en = ["hello world", "good morning", "how are you"] * 100
        dummy_fr = ["bonjour le monde", "bonjour", "comment ca va"] * 100
        with open(TRAIN_EN_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_en))
        with open(TRAIN_FR_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_fr))
        # Tạo luôn val/test giả để code không lỗi
        with open(VAL_EN_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_en[:10]))
        with open(VAL_FR_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_fr[:10]))
        with open(TEST_EN_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_en[:10]))
        with open(TEST_FR_PATH, 'w', encoding='utf-8') as f: f.write('\n'.join(dummy_fr[:10]))

    train_data = EnFrDataset(TRAIN_EN_PATH, TRAIN_FR_PATH)
    val_data = EnFrDataset(VAL_EN_PATH, VAL_FR_PATH)
    test_data = EnFrDataset(TEST_EN_PATH, TEST_FR_PATH)

    vocab_en, vocab_fr = build_vocabularies(train_data)
    collate_fn = create_collate_fn(vocab_en, vocab_fr)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, vocab_en, vocab_fr