# model.py  (overwrite existing)
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, embed_size=256, train_cnn=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove last fc
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # Use LayerNorm instead of BatchNorm to allow batch_size=1
        self.norm = nn.LayerNorm(embed_size)
        self.train_cnn = train_cnn

    def forward(self, images):
        # If train_cnn is False, freeze CNN gradients
        with torch.set_grad_enabled(self.train_cnn):
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.norm(features)
        return features  # (batch_size, embed_size)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, features, captions):
        # captions: (batch, seq_len)
        embeddings = self.embed(captions)
        # prepend features to embeddings: expand features to (batch, 1, embed_size)
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20):
        "Generate captions for given image features (greedy search)."
        sampled_ids = []
        inputs = features.unsqueeze(1)  # (1,1,embed)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (1,1,hidden)
            outputs = self.linear(hiddens.squeeze(1))    # (1,vocab_size)
            _, predicted = outputs.max(1)                # (1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)               # (1,embed_size)
            inputs = inputs.unsqueeze(1)                 # (1,1,embed_size)
        return sampled_ids
