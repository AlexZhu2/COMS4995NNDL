
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..annotations import COCOCaptionDataset  # Assume your dataset class is saved here
from ..model.caption_models.vanilla_CNN_transformer import CNNTransformerCaptioningModel

# ------------------------- Config -------------------------
BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 50
EMBED_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Dataset -------------------------
train_dataset = COCOCaptionDataset('captions_train2017.json', max_length=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

VOCAB_SIZE = len(train_dataset.vocab)
PAD_IDX = train_dataset.pad_idx

# ------------------------- Model -------------------------
model = CNNTransformerCaptioningModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN,
    dropout=DROPOUT,
    pad_idx=PAD_IDX
).to(DEVICE)

# ------------------------- Training -------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        images = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)

        tgt_input = input_ids[:, :-1]
        tgt_output = input_ids[:, 1:]

        logits = model(images, tgt_input)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} average loss: {total_loss / len(train_loader):.4f}")
