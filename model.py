import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) # Pretrained ResNet Model
        for param in resnet.parameters():
            param.requires_grad_(False)           # Since it is pretrained
        
        modules = list(resnet.children())[:-1]    # Get the last layer of ResNet
        self.resnet = nn.Sequential(*modules)     # Create embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0, 
                             batch_first=True
                           )
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]              # Discard the <end> word
        captions = self.embedding(captions) 
        '''
        outputs : (batch_size, caption length , embed_size)
        
        input/features : (batch_size, embed_size)
        inputs : (batch_size, caption length, embed_size)
        
        '''
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        outputs, _ = self.lstm(inputs)           # Get the output and hidden state
        outputs = self.linear(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        
        for ite in range(max_len+1):
            output, states = self.lstm(inputs,states)
            output = self.linear(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            end_index = 1
            if (predicted_index == end_index):
                break
            
            inputs = self.embedding(predicted_index)   
            inputs = inputs.unsqueeze(1)
        return outputs
        
        
        
        