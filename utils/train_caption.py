import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from annotations.COCO_caption_dataset import COCOCaptionDataset
from model.caption_models.vanilla_CNN_transformer import CNNTransformerCaptioningModel

# ------------------------- Config -------------------------
BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 50
EMBED_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Transforms -------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=10),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------------- Dataset -------------------------
train_dataset = COCOCaptionDataset('captions_train2017.json', image_root='train2017', max_length=MAX_LEN, transform=train_transform, vocab=None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

vocab = train_dataset.vocab

val_dataset = COCOCaptionDataset('captions_val2017.json', image_root='val2017', max_length=MAX_LEN, transform=val_transform, vocab=vocab)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_losses, val_losses = [], []
best_val_loss = float("inf")
val_sample_idx = 0  # fixed validation sample

os.makedirs("content/drive/MyDrive/Visualizations/", exist_ok=True)
os.makedirs("content/drive/MyDrive/Checkpoints/", exist_ok=True)
os.makedirs("content/drive/MyDrive/Stats/", exist_ok=True)
# ------------------------- Training Loop -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

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

        total_train_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    train_losses.append(total_train_loss / len(train_loader))

    # ------------------------- Validation -------------------------
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for i, batch in enumerate(val_progress):
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            tgt_input = input_ids[:, :-1]
            tgt_output = input_ids[:, 1:]

            logits = model(images, tgt_input)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            total_val_loss += loss.item()
            val_progress.set_postfix(loss=loss.item())

            if i == val_sample_idx:
                pred_ids = torch.argmax(logits[0], dim=-1).cpu()
                raw_img = batch['image'][0].cpu()
                raw_caption = batch['raw_caption'][0] if isinstance(batch['raw_caption'], list) else batch['raw_caption']
                pred_text = val_dataset.decode(pred_ids)

                img_np = raw_img.permute(1, 2, 0).numpy()
                std = np.array([0.229, 0.224, 0.225])
                mean = np.array([0.485, 0.456, 0.406])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)

                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                plt.axis("off")
                plt.title(f"GT: {raw_caption}\\nPred: {pred_text}", fontsize=10)
                plt.savefig(f"/content/drive/MyDrive/Visualizations/epoch_{epoch+1}.png")
                plt.close()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "/cotent/drive/MyDrive/Checkpoints/best_model_vanilla_CNN_transformer.pth")
        print("✅ Saved best model.")

# ------------------------- Plot Loss Curve -------------------------
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.savefig("/content/drive/MyDrive/Stats/loss_curve.png")