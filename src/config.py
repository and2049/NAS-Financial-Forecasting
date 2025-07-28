import os

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Configuration ---
TICKER_SYMBOL = 'SPY'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed', f'{TICKER_SYMBOL}_processed.csv')

# --- Feature Configuration ---
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal',
    'ATR', 'ROC'
]
TARGET_VARIABLE = 'Target'
SEQUENCE_LENGTH = 10

# --- Model & Search Configuration ---
INPUT_CHANNELS = len(FEATURES)
INIT_CHANNELS = 16
N_CELLS = 4
N_NODES = 4
PRIMITIVES = [
    'none', 'max_pool_3x1', 'avg_pool_3x1', 'skip_connect',
    'sep_conv_3x1', 'sep_conv_5x1', 'dil_conv_3x1', 'dil_conv_5x1'
]

# --- Search Training Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.025
MOMENTUM = 0.9
WEIGHT_DECAY = 3e-4
EPOCHS = 100

# --- Final Model Training Configuration ---
FINAL_LEARNING_RATE = 0.001
FINAL_WEIGHT_DECAY = 1e-3
FINAL_EPOCHS = 50
DROPOUT_RATE = 0.5
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 15
USE_WEIGHTED_LOSS = True
MINORITY_CLASS_WEIGHT_MULTIPLIER = 1.7
