# Image Captioning with CNN and Transformer Decoder (BART)

This repository contains the implementation of a CNN-Transformer architecture for image captioning, combining a CNN-based visual encoder with a BART-based language decoder, along with attention map visualization and analysis of cross-attention between image regions and generated tokens.

## Annotations
Annotation json files for both training and validation set can be found at [Here](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). We will retrieve images via Internet, no need to download the actual image dataset.

## Features

- **CNN Encoders**: ResNet-50 and EfficientNetV2 S for extracting spatial visual features
- **Transformer Decoder**: Pretrained BART-base model for fluent and contextually relevant caption generation
- **Attention Map Visualization**: Tools to extract and plot cross-attention weights between image features and text tokens
- **Evaluation**: Quantitative metrics (BLEU-4, METEOR) and qualitative examples on the COCO 2017 Captioning dataset

## Repository Structure

```
COMS4995NNDL/
├── annotations/
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── COCO_caption_dataset.py
│   └── COCO_caption_dataset_bart.py
├── model/
│   ├── encoder/
│   │   └── CNN_encoder.py
│   ├── decoder/
│   │   └── transformer_decoder_layer.py
│   └── caption_models/
│       └── bart_CNN_transformer.py
├── utils/
│   ├── .ipynb_checkpoints/
│   ├── Colab_continue_training_bart.ipynb
│   ├── Colab_training_bart.ipynb
│   ├── Colab_training.ipynb
│   ├── train_bart_caption.py
│   └── train_caption.py
└── readme.md
```

## Installation

**Clone the repository** and checkout the `BART` branch:

   ```bash
   git clone https://github.com/AlexZhu2/COMS4995NNDL.git
   cd COMS4995NNDL
   git checkout BART
   ```

### 1. Training

These blocks run under `model.train()` and update model parameters:

```python
# Model & Optimization Setup
model = CNNBARTCaptioningModel(embed_dim=EMBED_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Inside the training loop
model.train()
for batch in train_loader:
    images, captions = ...
    logits = model(images, tgt_input, decoder_attention_mask=attention_mask)
    loss = criterion(
        logits.reshape(-1, VOCAB_SIZE),
        tgt_output.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After epoch
scheduler.step()
```
You can swap `resnet50` with `efficientnet_v2_s` for improved performance.

### 2. Validation

These blocks run under `model.eval()` without gradient updates to compute validation loss and save checkpoints:

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        images, captions = ...
        logits = model(images, tgt_input, decoder_attention_mask=attention_mask)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_output.reshape(-1)
        )
        total_val_loss += loss.item()
```

### 3. Inference (Qualitative Sampling)

Part of the validation loop that decodes and visualizes sample predictions:

```python
# Inside validation loop, for selected indices
pred_ids = torch.argmax(logits[0], dim=-1).cpu().tolist()
pred_text = val_dataset.decode(pred_ids)

# Denormalize image and plot
plt.imshow(denormalized_image); plt.axis('off')
plt.title(f"GT: {raw_caption}\nPred: {pred_text}")
plt.savefig("...png")
```

This script outputs predicted captions along with PNG files visualizing cross-attention overlays on the input image.

## Evaluation

Quantitative evaluation on the COCO validation set can be run via:

```bash
python evaluate.py \
  --predictions outputs/predictions.json \
  --references /path/to/coco/annotations/captions_val2017.json
```

Metrics reported: BLEU-4 and METEOR.

## Results

- **EfficientNetV2 S + BART**: BLEU-4 = 0.2772, METEOR = 0.4298
- **ResNet-50 + BART**: BLEU-4 = 0.2686, METEOR = 0.4260