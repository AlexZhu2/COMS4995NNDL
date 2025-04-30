
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import random
from torchvision import transforms
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Iterator
from io import BytesIO
import requests

class COCOCaptionDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None, max_length=50):
        '''
        Args:
            json_path (str): Path to the COCO captions JSON file
            image_root (str): Directory containing the COCO images (e.g. 'train2017/')
            transform (callable, optional): Transform to apply to images
            max_length (int): Maximum tokenized caption length
        '''
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.max_length = max_length
        self.tokenizer = get_tokenizer('basic_english')

        # Map image_id to metadata + captions
        self.image_dict = {}
        for img in self.data['images']:
            self.image_dict[img['id']] = {
                'file_name': img['file_name'],
                'flickr_url': img.get('flickr_url', None),
                'captions': []
            }

        all_captions = []
        for ann in self.data['annotations']:
            caption = ann['caption']
            self.image_dict[ann['image_id']]['captions'].append(caption)
            all_captions.append(caption)

        print("Building vocabulary...")
        self.vocab = self._build_vocabulary(all_captions)
        print(f"Vocabulary size: {len(self.vocab)}")

        self.image_ids = list(self.image_dict.keys())

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.pad_idx = self.vocab['<pad>']
        self.unk_idx = self.vocab['<unk>']
        self.start_idx = self.vocab['<start>']
        self.end_idx = self.vocab['<end>']

    def _yield_tokens(self, data_iter: List[str]) -> Iterator[List[str]]:
        for text in data_iter:
            yield self.tokenizer(text.lower())

    def _build_vocabulary(self, texts: List[str]):
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        vocab = build_vocab_from_iterator(
            self._yield_tokens(texts),
            min_freq=1,
            specials=special_tokens,
            special_first=True
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def encode(self, text: str) -> torch.Tensor:
        tokens = ['<start>'] + self.tokenizer(text.lower()) + ['<end>']
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
        indices = self.vocab(tokens)
        tensor = torch.tensor(indices)
        padded = torch.full((self.max_length,), self.pad_idx, dtype=torch.long)
        padded[:len(tensor)] = tensor
        return padded

    def decode(self, indices: torch.Tensor) -> str:
        tokens = [self.vocab.get_itos()[idx.item()] for idx in indices if idx.item() != self.pad_idx]
        tokens = [token for token in tokens if token not in ['<start>', '<end>']]
        return ' '.join(tokens)

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
