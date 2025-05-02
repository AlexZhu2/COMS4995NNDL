import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from annotations.COCO_caption_dataset_bart import COCOCaptionDataset
from model.caption_models.bart_CNN_transformer import CNNBARTCaptioningModel

# ------------------------- Config -------------------------
BATCH_SIZE = 32
EPOCHS     = 10
MAX_LEN    = 35
EMBED_DIM  = 512    # must match your CNNEncoder output dim
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Transforms -------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=10),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ------------------------- Dataset -------------------------
train_dataset = COCOCaptionDataset(
    'COMS4995NNDL/annotations/captions_train2017.json',
    image_root='train2017',
    max_length=MAX_LEN,
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = COCOCaptionDataset(
    'COMS4995NNDL/annotations/captions_val2017.json',
    image_root='val2017',
    max_length=MAX_LEN,
    transform=val_transform,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

VOCAB_SIZE = train_dataset.tokenizer.vocab_size
PAD_IDX    = train_dataset.pad_idx

# ------------------------- Model -------------------------
model = CNNBARTCaptioningModel(
    pretrained_model_name="facebook/bart-base",
    embed_dim=EMBED_DIM,
    freeze_encoder=True
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_losses, val_losses = [], []
best_val_loss = float("inf")
val_sample_idx = [0,10,246,93,59]  # fixed validation samples

os.makedirs("/content/drive/MyDrive/Visualizations-BART/", exist_ok=True)
os.makedirs("/content/drive/MyDrive/Checkpoints-BART/", exist_ok=True)
os.makedirs("/content/drive/MyDrive/Stats-BART/", exist_ok=True)

# ------------------------- Training Loop -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for batch in train_pbar:
        images    = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)

        tgt_input  = input_ids[:, :-1]
        tgt_output = input_ids[:, 1:]

        decoder_attention_mask = (tgt_input != PAD_IDX)

        logits = model(images, tgt_input, decoder_attention_mask=decoder_attention_mask)  # (B, T, V)
        loss   = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_pbar.set_postfix(loss=loss.item())

    train_losses.append(total_train_loss / len(train_loader))

    # ------------------------- Validation -------------------------
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for i, batch in enumerate(val_pbar):
            images    = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)

            tgt_input  = input_ids[:, :-1]
            tgt_output = input_ids[:, 1:]

            logits = model(images, tgt_input)
            loss   = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt_output.reshape(-1)
            )
            total_val_loss += loss.item()
            val_pbar.set_postfix(loss=loss.item())

            # save one sample visualization per epoch
            global_idx_start = i * val_loader.batch_size

            for j in range(batch['image'].size(0)):
                global_idx = global_idx_start + j
                if global_idx in val_sample_idx:
                    pred_ids = torch.argmax(logits[j], dim=-1).cpu()
                    raw_img = batch['image'][j].cpu()
                    raw_caption = (batch['raw_caption'][j]
                                  if isinstance(batch['raw_caption'], list)
                                  else batch['raw_caption'])
                    pred_text = val_dataset.decode(pred_ids)

                    img_np = raw_img.permute(1, 2, 0).numpy()
                    std = np.array([0.229, 0.224, 0.225])
                    mean = np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np * std + mean, 0, 1)

                    plt.figure(figsize=(6,6))
                    plt.imshow(img_np)
                    plt.axis("off")
                    plt.title(f"GT: {raw_caption}\nPred: {pred_text}", fontsize=10)
                    with open(f"/content/drive/MyDrive/Visualizations-BART/epoch_{epoch+1}.txt", "a") as f:
                        f.write(f"Validation Sample {global_idx}: GT: {raw_caption}\nPred: {pred_text}\n\n")
                    plt.savefig(f"/content/drive/MyDrive/Visualizations-BART/epoch_{epoch+1}_sample_{global_idx}.png")
                    plt.close()


    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

# -------------- Save loss curve after each epoch --------------
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"/content/drive/MyDrive/Stats-BART/loss_curve_epoch_{epoch+1}.png")
    plt.close()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(
            model.state_dict(),
            "/content/drive/MyDrive/Checkpoints-BART/best_model_cnn_bart_captioning.pth"
        )
        print("✅ Saved best model.")