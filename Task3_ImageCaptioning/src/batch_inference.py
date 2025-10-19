# batch_inference.py
import os
import argparse
import csv
import torch
from inference import load_models, generate_caption

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, 'predictions.csv')

    # load encoder, decoder, vocab
    encoder, decoder, vocab = load_models(args.checkpoint, args.vocab_file, device)

    # gather all image files
    images = [f for f in os.listdir(args.img_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(images)} images in {args.img_dir}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'predicted_caption'])

        for img_name in images:
            img_path = os.path.join(args.img_dir, img_name)
            caption = generate_caption(img_path, encoder, decoder, vocab, device, max_len=args.max_len)
            writer.writerow([img_name, caption])
            print(f"{img_name}  -->  {caption}")

    print(f"\n✅ Captions saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, default='../data/vocab.pkl')
    parser.add_argument('--img_dir', type=str, default='../data/images')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--max_len', type=int, default=20)
    args = parser.parse_args()
    main(args)
