import torch
from torch.utils.data import Dataset
import json
import requests
from PIL import Image
from io import BytesIO
import random
from torchvision import transforms
import numpy as np
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import List, Iterator

class COCOCaptionDataset(Dataset):
    def __init__(self, json_path, transform=None, max_length=50):
        """
        Args:
            json_path (string): Path to the COCO captions JSON file
            transform (callable, optional): Optional transform to be applied on the image
            max_length (int): Maximum length of the tokenized caption
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.max_length = max_length
        # Initialize the tokenizer (using basic_english tokenizer)
        self.tokenizer = get_tokenizer('basic_english')
        
        # Create a dictionary mapping image_id to its metadata and captions
        self.image_dict = {}
        for img in self.data['images']:
            self.image_dict[img['id']] = {
                'flickr_url': img['flickr_url'],
                'captions': []
            }
        
        # Add captions to the image dictionary and collect all captions
        all_captions = []
        for ann in self.data['annotations']:
            caption = ann['caption']
            self.image_dict[ann['image_id']]['captions'].append(caption)
            all_captions.append(caption)
        
        # Build vocabulary
        print("Building vocabulary...")
        self.vocab = self._build_vocabulary(all_captions)
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Convert dictionary to list for indexing
        self.image_ids = list(self.image_dict.keys())
        
        # Default transform if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization values
        ])
        
        # Special tokens
        self.pad_idx = self.vocab['<pad>']
        self.unk_idx = self.vocab['<unk>']
        self.start_idx = self.vocab['<start>']
        self.end_idx = self.vocab['<end>']

    def _yield_tokens(self, data_iter: List[str]) -> Iterator[List[str]]:
        """Tokenize captions and yield tokens"""
        for text in data_iter:
            yield self.tokenizer(text.lower())

    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Add special tokens
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
        """Convert text to sequence of indices"""
        # Tokenize and add special tokens
        tokens = ['<start>'] + self.tokenizer(text.lower()) + ['<end>']
        
        # Truncate if necessary
        if len(tokens) > self.max_length - 1:
            tokens = tokens[:self.max_length - 1]
            
        # Convert to indices
        indices = self.vocab(tokens)
        
        # Convert to tensor and pad
        tensor = torch.tensor(indices)
        padded = torch.full((self.max_length,), self.pad_idx, dtype=torch.long)
        padded[:len(tensor)] = tensor
        
        return padded

    def decode(self, indices: torch.Tensor) -> str:
        """Convert sequence of indices back to text"""
        # Remove padding
        tokens = [self.vocab.get_itos()[idx.item()] for idx in indices if idx.item() != self.pad_idx]
        # Remove special tokens
        tokens = [token for token in tokens if token not in ['<start>', '<end>']]
        return ' '.join(tokens)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_dict[image_id]
        
        # Fetch image from URL
        try:
            response = requests.get(image_info['flickr_url'])
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get random caption and tokenize it
        captions = image_info['captions']
        caption = random.choice(captions) if captions else ""
        
        # Tokenize the caption
        input_ids = self.encode(caption)
        # print("\nTokenization example:")
        # print(f"Original caption: {caption}")
        # print(f"Tokenized indices: {input_ids}")
        # print(f"Decoded tokens: {[self.vocab.get_itos()[idx.item()] for idx in input_ids]}")
        # print(f"Decoded caption (without special tokens): {self.decode(input_ids)}")
        attention_mask = (input_ids != self.pad_idx).float()
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'raw_caption': caption
        }

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Path to the COCO captions JSON file
    json_path = 'captions_train2017.json'
    dataset = COCOCaptionDataset(json_path)
    
    # Print dataset statistics
    # print(f"Dataset size: {len(dataset)}")
    
    # Get one sample
    sample = dataset[0]
    # print(f"Image shape: {sample['image'].shape}")
    # print(f"Input IDs shape: {sample['input_ids'].shape}")
    # print(f"Attention mask shape: {sample['attention_mask'].shape}")
    # print(f"Attention mask: {sample['attention_mask']}")
    # print(f"\nRaw caption: {sample['raw_caption']}")
    # print(f"Tokenized and decoded caption: {dataset.decode(sample['input_ids'])}")
    # print(f"Vocabulary size: {len(dataset.vocab)}")
    
    # Visualize the image
    image = sample['image']
    
    # Correct denormalization
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image * std + mean
    
    # Convert to numpy and transpose for visualization
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)  # Clip values to valid range
    
    plt.imshow(image)
    plt.title(sample['raw_caption'])
    plt.show()

