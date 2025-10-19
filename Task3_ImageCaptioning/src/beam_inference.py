# beam_inference.py
import torch
from model import EncoderCNN, DecoderRNN
import pickle
from PIL import Image
from torchvision import transforms
import argparse
import os
from inference import load_vocab, load_models  # if you used the robust inference.py earlier
import math

def beam_search(encoder, decoder, vocab, image_path, device, beam_size=3, max_len=20):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        features = encoder(image_tensor)  # (1,embed)
    # initialize beam
    k = beam_size
    vocab_stoi = vocab.stoi
    vocab_itos = vocab.itos
    # Each beam: (sequence_of_ids, log_prob, hidden_states)
    # Start with <SOS>
    start_token = vocab_stoi.get("<SOS>", 1)
    end_token = vocab_stoi.get("<EOS>", 2)

    # initial input to LSTM: features as first input
    # We'll operate by repeatedly expanding beams
    sequences = [[ [start_token], 0.0, None ]]  # list of [seq, score, states]

    for _ in range(max_len):
        all_candidates = []
        for seq, score, states in sequences:
            if seq[-1] == end_token:
                all_candidates.append((seq, score, states))
                continue
            # prepare input: last token embedding if len(seq)>1 else features
            if len(seq)==1:
                inputs = features.unsqueeze(1)  # (1,1,embed)
            else:
                last_id = torch.tensor([seq[-1]]).to(device)
                emb = decoder.embed(last_id).unsqueeze(1)  # (1,1,embed)
                inputs = emb
            # run one LSTM step
            hiddens, new_states = decoder.lstm(inputs, states) if states is not None else decoder.lstm(inputs)
            outputs = decoder.linear(hiddens.squeeze(1))  # (1,vocab)
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1).squeeze(0)  # (vocab,)
            topk_logps, topk_ids = torch.topk(log_probs, k)
            for i in range(k):
                candidate_seq = seq + [int(topk_ids[i].item())]
                candidate_score = score + float(topk_logps[i].item())
                all_candidates.append((candidate_seq, candidate_score, new_states))
        # pick k best
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:k]
        # if all sequences end with EOS, stop
        if all(s[0][-1]==end_token for s in sequences):
            break

    # pick the best sequence
    best_seq = sequences[0][0]
    # convert ids to words until EOS (skip SOS)
    words = []
    for idx in best_seq:
        if idx==start_token: continue
        if idx==end_token: break
        words.append(vocab_itos.get(idx, "<UNK>"))
    return " ".join(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--beam_size', type=int, default=3)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, vocab = load_models(args.checkpoint, args.vocab_file, device)
    out = beam_search(encoder, decoder, vocab, args.image, device, beam_size=args.beam_size)
    import os
image_name = os.path.basename(args.image)
print(f"Image: {image_name}  -->  Caption: {out}")

