# train.py  (overwrite your existing file)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CaptionDataset
from model import EncoderCNN, DecoderRNN
import argparse
import os
from utils import save_checkpoint
import pickle
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def collate_fn(batch):
    images = []
    captions = []
    for img, cap, _ in batch:
        images.append(img)
        captions.append(cap)
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

def load_vocab_obj(vocab_file):
    with open(vocab_file, 'rb') as f:
        data = pickle.load(f)

    # If it's a dict with 'stoi'/'itos', wrap it
    if isinstance(data, dict) and 'stoi' in data and 'itos' in data:
        class VocabWrapper:
            def __init__(self, itos, stoi):
                self.itos = itos
                self.stoi = stoi
            def numericalize(self, text):
                return []
        return VocabWrapper(data['itos'], data['stoi'])
    # If it's an object with attributes, return as-is
    if hasattr(data, 'stoi') and hasattr(data, 'itos'):
        return data
    # Unknown format
    raise RuntimeError("Unrecognized vocab format in {}. Expected dict with 'stoi'/'itos' or object with attributes.".format(vocab_file))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = CaptionDataset(args.caption_file, args.img_dir, args.vocab_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # load vocab wrapper so we can get stoi/itos
    vocab = load_vocab_obj(args.vocab_file)

    vocab_size = len(vocab.itos)
    embed_size = args.embed_size
    hidden_size = args.hidden_size

    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    # Use PAD index if available, otherwise default to 0
    pad_index = vocab.stoi.get("<PAD>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    # Collect parameters: decoder always trainable; for encoder only train linear and norm (or bn) if present
    params = list(decoder.parameters())
    # encoder linear params
    if hasattr(encoder, 'linear'):
        params += list(encoder.linear.parameters())
    # encoder normalization params (support bn or norm)
    if hasattr(encoder, 'bn'):
        params += list(encoder.bn.parameters())
    elif hasattr(encoder, 'norm'):
        params += list(encoder.norm.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        steps = 0
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions)

            # flatten outputs and targets
            outputs = outputs.view(-1, outputs.size(2))
            targets = captions.view(-1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            loop.set_postfix(loss=running_loss / steps)

        # Save checkpoint at end of epoch (save encoder + decoder + optimizer)
        os.makedirs(args.model_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': decoder.state_dict(),
            'encoder_dict': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'vocab_size': vocab_size
        }
        save_checkpoint(checkpoint, filename=os.path.join(args.model_dir, f'checkpoint_epoch{epoch+1}.pth.tar'))

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='../data/captions.csv')
    parser.add_argument('--img_dir', type=str, default='../data/images')
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--model_dir', type=str, default='../models')
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
