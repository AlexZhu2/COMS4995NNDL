import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random
from torchvision import transforms
from typing import Optional
from io import BytesIO
import requests
from transformers import BartTokenizer

class COCOCaptionDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: str,
        transform=None,
        max_length: int = 35,
        tokenizer: Optional[BartTokenizer] = None
    ):
        """
        Args:
            json_path (str): path to COCO captions JSON
            image_root (str): image directory
            transform (callable): image transform
            max_length (int): max token length
            tokenizer (BartTokenizer): HuggingFace tokenizer
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.max_length = max_length

        # 1️⃣ use pretrained tokenizer (or load if None)
        self.tokenizer = tokenizer or BartTokenizer.from_pretrained("facebook/bart-base")

        # 2️⃣ map image_id → metadata
        self.image_dict = {}
        for img in self.data['images']:
            self.image_dict[img['id']] = {
                'file_name': img['file_name'],
                'flickr_url': img.get('flickr_url', None),
                'captions': []
            }

        for ann in self.data['annotations']:
            caption = ann['caption']
            self.image_dict[ann['image_id']]['captions'].append(caption)

        self.image_ids = list(self.image_dict.keys())

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # token special indices
        self.pad_idx = self.tokenizer.pad_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.bos_idx = self.tokenizer.bos_token_id

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenize caption → padded tensor
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding.input_ids.squeeze(0)  # shape: (max_length,)

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode token IDs → string
        """
        return self.tokenizer.decode(
            indices.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.image_dict[image_id]

        img_path = os.path.join(self.image_root, info['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load local image {img_path}, trying URL...")
            if info.get('flickr_url'):
                try:
                    response = requests.get(info['flickr_url'])
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    print(f"Failed to load from URL: {info['flickr_url']}")
                    image = Image.new('RGB', (224, 224))
            else:
                image = Image.new('RGB', (224, 224))

        image = self.transform(image)

        caption = random.choice(info['captions']) if info['captions'] else ""
        input_ids = self.encode(caption)
        attention_mask = (input_ids != self.pad_idx).float()

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'raw_caption': caption
        }
