# inference.py  (overwrite existing)
import torch
from model import EncoderCNN, DecoderRNN
import pickle
from PIL import Image
from torchvision import transforms
import argparse
import os
import sys

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        data = pickle.load(f)

    # If dict with 'itos'/'stoi'
    if isinstance(data, dict) and 'itos' in data and 'stoi' in data:
        class VocabWrapper:
            def __init__(self, itos, stoi):
                self.itos = itos
                self.stoi = stoi
        return VocabWrapper(data['itos'], data['stoi'])

    # If object with attributes
    if hasattr(data, 'itos') and hasattr(data, 'stoi'):
        return data

    # Try to infer simple mapping if it's e.g. a plain list/dict
    raise RuntimeError("Unrecognized vocab format in {}. Expected a dict with 'itos'/'stoi' or an object with those attributes.".format(vocab_path))

def load_models(checkpoint_path, vocab_path, device):
    vocab = load_vocab(vocab_path)

    # compute vocab size
    try:
        vocab_size = len(vocab.itos)
    except Exception:
        vocab_size = None

    # load checkpoint first (to get checkpoint vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_vocab_size = checkpoint.get('vocab_size', None)

    # If both exist and mismatch, raise helpful error
    if ckpt_vocab_size is not None and vocab_size is not None and ckpt_vocab_size != vocab_size:
        raise RuntimeError(
            f"Vocabulary size mismatch: checkpoint has vocab_size={ckpt_vocab_size} but vocab file has {vocab_size}.\n"
            f"Possible fixes:\n"
            f" - Make sure you pass the same --vocab_file that was used during training.\n"
            f" - If you re-built vocab after training, re-run training or use the original vocab.pkl saved with the model.\n"
            f"Checkpoint path: {checkpoint_path}\nVocab path: {vocab_path}"
        )

    # If ckpt_vocab_size exists but vocab_size is None, use checkpoint value
    if vocab_size is None and ckpt_vocab_size is not None:
        vocab_size = ckpt_vocab_size

    if vocab_size is None:
        raise RuntimeError("Unable to determine vocabulary size from vocab file or checkpoint.")

    embed_size = 256
    hidden_size = 512

    encoder = EncoderCNN(embed_size=embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # load parameters safely
    if 'encoder_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_dict'])
    elif 'encoder_state' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state'])
    if 'state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['state_dict'])
    elif 'decoder_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_dict'])
    else:
        # try loading entire checkpoint directly (rare)
        try:
            decoder.load_state_dict(checkpoint)
        except Exception:
            pass

    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder, vocab

def generate_caption(image_path, encoder, decoder, vocab, device, max_len=20):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features, max_len=max_len)
    # convert to words until EOS
    words = []
    for idx in sampled_ids:
        if idx == vocab.stoi.get("<EOS>"):
            break
        word = vocab.itos.get(idx, "<UNK>")
        words.append(word)
    caption = " ".join(words)
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        encoder, decoder, vocab = load_models(args.checkpoint, args.vocab_file, device)
    except Exception as e:
        print("ERROR loading models:", e)
        sys.exit(1)

    caption = generate_caption(args.image, encoder, decoder, vocab, device)
    print("Caption:", caption)
