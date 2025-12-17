import torch

# Đường dẫn file dữ liệu 
TRAIN_EN_PATH = 'Data/train/train.en'
TRAIN_FR_PATH = 'Data/train/train.fr'
VAL_EN_PATH = 'Data/val/val.en'
VAL_FR_PATH = 'Data/val/val.fr'
TEST_EN_PATH = 'Data/test/test2017/test_2017_mscoco.en'
TEST_FR_PATH = 'Data/test/test2017/test_2017_mscoco.fr'

# Cấu hình Model 
INPUT_DIM = 10000  
OUTPUT_DIM = 10000
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256     
N_LAYERS = 2      
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Huấn luyện
BATCH_SIZE = 64
N_EPOCHS = 30      
LEARNING_RATE = 0.001
CLIP = 1           # Gradient clipping

# Thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = 'models/best_model.pth'