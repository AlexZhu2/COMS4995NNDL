import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CocoCaptionDataset(Dataset):
    def __init__(self, img_folder, ann_path, vocab, transform=None, max_len=20):
        self.img_folder = img_folder
        self.max_len = max_len
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        with open(ann_path, 'r') as f:
            data = json.load(f)

        # Map image IDs to file names
        self.id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        self.samples = [(ann["image_id"], ann["caption"]) for ann in data["annotations"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, caption = self.samples[idx]
        img_path = os.path.join(self.img_folder, self.id_to_filename[image_id])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_ids = torch.tensor(self.vocab.numericalize(caption, self.max_len))
        return image, caption_ids
