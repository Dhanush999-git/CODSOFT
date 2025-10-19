# evaluate.py
import argparse
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import os

nltk.download('punkt')

def simple_tokenize(s):
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", "", s)
    return s.split()

def main(args):
    preds = pd.read_csv(args.pred_csv)
    refs = pd.read_csv(args.caption_file)
    # make dict: image -> caption
    ref_map = {r['image_name']: r['caption'] for _, r in refs.iterrows()}

    smoothie = SmoothingFunction().method4
    rows = []

    for _, r in preds.iterrows():
        img = r['image_name']
        pred = str(r['predicted_caption'])
        ref = ref_map.get(img, "")
        pred_tokens = simple_tokenize(pred)
        ref_tokens = simple_tokenize(ref)
        if len(pred_tokens)==0: 
            score = 0.0
        else:
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        rows.append((img, ref, pred, score))
        print(f"{img}  BLEU={score:.4f}\n  ref: {ref}\n  pred: {pred}\n")

    df_out = pd.DataFrame(rows, columns=['image','reference','prediction','bleu'])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print("Average BLEU:", df_out['bleu'].mean())
    print("Saved evaluation to", args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_csv', type=str, default='../results/predictions.csv')
    parser.add_argument('--caption_file', type=str, default='../data/captions.csv')
    parser.add_argument('--output', type=str, default='../results/eval_results.csv')
    args = parser.parse_args()
    main(args)
