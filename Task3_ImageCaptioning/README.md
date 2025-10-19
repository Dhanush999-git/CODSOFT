# CODSOFT - Image Captioning (Task 3)

This repository contains a simple image captioning implementation using a pretrained ResNet encoder and an LSTM decoder.

## Summary
- Task: Image Captioning (Task 3 from CODSOFT internship tasks). See uploaded instructions for task descriptions.
- Approach: Pretrained ResNet-50 extracts image features; an LSTM decoder generates captions token-by-token.
- Purpose: Educational/demo implementation suitable for a short internship project and video demo.

## Project structure
(see project structure above)

## Quick start

1. Prepare data:
   - Provide a CSV `captions.csv` with two columns: `image_file`, `caption`.
   - Place images under `data/images/` and set `--root data/images` when training.

   Example CSV:
