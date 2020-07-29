import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                            batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        features = features.view(features.size(0), 1, -1)
        embed = self.word_embed(captions[:,:-1])
        inputs = torch.cat((features, embed), dim=1)
        decode,_ = self.lstm(inputs)
        decode = self.linear(decode)
        
        return decode

    def sample(self, inputs, states=None, max_len=20,stop_idx=1):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out_list = []
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.linear(outputs.squeeze(1))
            target_idx = outputs.max(1)[1]
            out_list.append(target_idx.item())
            if target_idx == stop_idx:
                break
            inputs = self.word_embed(target_idx).unsqueeze(1)
        return out_list
