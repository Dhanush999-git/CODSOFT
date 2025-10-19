# build_vocab.py  (overwrite existing)
import nltk
import pickle
import argparse
from collections import Counter
import pandas as pd
import os
import re

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9'\s]", "", text)
        return nltk.tokenize.word_tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(tok, self.stoi["<UNK>"])
            for tok in tokenized_text
        ]

def main(args):
    df = pd.read_csv(args.caption_file)
    captions = df['caption'].tolist()
    vocab = Vocabulary(freq_threshold=args.threshold)
    vocab.build_vocabulary(captions)

    # Save plain dict so pickle doesn't need class when loading
    vocab_data = {
        'itos': vocab.itos,
        'stoi': vocab.stoi,
        'freq_threshold': vocab.freq_threshold
    }

    with open(args.vocab_file, 'wb') as f:
        pickle.dump(vocab_data, f)

    print(f"Vocabulary saved to {args.vocab_file}, size: {len(vocab)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='../data/captions.csv')
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--threshold', type=int, default=1)
    args = parser.parse_args()
    main(args)
