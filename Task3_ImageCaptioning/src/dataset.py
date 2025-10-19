# dataset.py  (overwrite existing)
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import pickle

class SimpleVocabWrapper:
    """A tiny wrapper providing .stoi, .itos and numericalize using tokenization similar to original."""
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    @staticmethod
    def tokenizer_eng(text):
        import re, nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            nltk.download('punkt')
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", "", text)
        return nltk.tokenize.word_tokenize(text)

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [ self.stoi.get(tok, self.stoi.get("<UNK>", 3)) for tok in tokenized_text ]

class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_dir, vocab_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # robustly load vocab: support old pickled class and new dict form
        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)

        # If data is a dict with 'itos'/'stoi' keys (new format), wrap it
        if isinstance(data, dict) and 'itos' in data and 'stoi' in data:
            self.vocab = SimpleVocabWrapper(data['itos'], data['stoi'])
        else:
            # Assume it's an object with attributes .itos and .stoi (older format)
            # This covers the case where the pickled object was an instance of Vocabulary class
            try:
                itos = getattr(data, 'itos')
                stoi = getattr(data, 'stoi')
                self.vocab = SimpleVocabWrapper(itos, stoi)
            except Exception as e:
                raise RuntimeError("Unable to load vocabulary from file. Make sure vocab.pkl "
                                   "is either a dict with 'itos'/'stoi' or a pickled Vocabulary instance.") from e

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        caption = row['caption']

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # numericalize caption and add <SOS> and <EOS>
        numericalized = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        caption_tensor = torch.tensor(numericalized)

        return image, caption_tensor, img_name
