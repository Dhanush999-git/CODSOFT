# 🖼️ Task 3: Image Captioning

## Project Overview
This project is part of the CODSOFT Artificial Intelligence Internship.

In this task, I built an **Image Captioning Model** that combines Computer Vision and Natural Language Processing (NLP) to automatically generate meaningful captions for images.

The model first extracts **visual features** from images using a pre-trained **CNN model** (like VGG16 or ResNet), and then uses a **Recurrent Neural Network (RNN)** or **LSTM-based model** to generate descriptive captions. This project demonstrates the integration of deep learning models from two major AI domains — **vision and language**. 🤖

---

## Features
- Extracts image features using a pre-trained **CNN (VGG16 / ResNet)**
- Generates captions using **RNN or LSTM models**
- Supports training on **custom datasets**
- Can generate captions for new, unseen images
- Produces **human-like captions** using learned language patterns

---

## Technologies Used
- **Python 3**
- **TensorFlow / PyTorch**
- **NumPy, Pandas**
- **NLTK** (for text preprocessing)
- **Matplotlib** (for visualization)
- **Pre-trained CNN (VGG16 / ResNet)**

## Project Structure
Task3_ImageCaptioning/
│── data/
│   ├── images/              # Folder containing training images
│   ├── captions.csv         # Image names and captions
│   └── vocab.pkl            # Serialized vocabulary file
│
│── models/
│   ├── checkpoint_epoch5.pth.tar  # Saved model checkpoint
│
│── results/
│   └── generated_captions.txt     # Output captions
│
│── src/
│   ├── build_vocab.py       # Builds vocabulary from captions
│   ├── train.py             # Training script
│   ├── model.py             # CNN + RNN model definition
│   ├── batch_inference.py   # Generate captions in batches
│   └── utils.py             # Helper functions
│
│── requirements.txt         # Dependencies
│── README.md                # Documentation (this file)

---

## How to Run
1.  Clone this repository or download the folder.
2.  Navigate to the project directory:
   cd Task3_ImageCaptioning
3.  Prepare your dataset:
    -   Place images in the data/images/ folder
    -   Create a data/captions.csv file with columns: image_name,caption (e.g., dog.jpg,A dog playing in the park)
4.  Build the vocabulary:
    python src/build_vocab.py --caption_file data/captions.csv --vocab_file data/vocab.pkl --threshold 1
5.  Train the model:
    python src/train.py
6.  Generate captions for new images:
    python src/batch_inference.py --checkpoint models/checkpoint_epoch5.pth.tar --vocab_file data/vocab.pkl --img_dir data/images --output_dir results

---

## Example Output
Input Image:
(An image of a dog playing on grass)

Generated Caption:

"A brown dog playing in the field."

---

## Future Improvements
- Experiment with **Transformer-based captioning models** (e.g., ViT + GPT) to leverage their superior sequential processing and attention mechanisms.
- Add **beam search** for better caption generation by exploring multiple high-probability sequences.
- Use **attention mechanisms** within the RNN/LSTM decoder for image–word alignment, allowing the model to focus on relevant image regions when generating specific words.
- Integrate with **Flask or Streamlit** for a web-based demo. 🌐
---
