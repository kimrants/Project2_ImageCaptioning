import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #here we took the resnet model which is pretrained
        for param in resnet.parameters(): # here we checked for its parameters and set them to requires_grad = false ,because those parameters are pretrained,we donot need to calculate gradient on those parameters. 
            param.requires_grad_(False) 
        
        modules = list(resnet.children())[:-1] # here we clipped the resnet's last fully connected layer so we'll end up with its feature vector that hasn't been flattened yet 
        self.resnet = nn.Sequential(*modules) # now we take those modules and convert them into sequential layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) # that sequential layer is now being converted to linear/fc layer which has the same dimensionality as embed_size, this is done so that we donot encounter any size mismatch while dealing with rnn later on


    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features  

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        
        # define the attributes that are taken in 
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        
        # the linear layer that maps the hidden state output dimension 
        # to the vocab_size
        self.hidden2vocab = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
    def forward(self, features, captions):
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        #captions = captions[:, :-1] 
        
        # create embedded word vectors for each token in a batch of captions
        embeddings = self.embed(captions)  

        # -> batch_size, caption (sequence) length, embed_size
        embeddings = torch.cat((features.unsqueeze(dim=1),embeddings), dim=1)
        
         # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, hidden = self.lstm(embeddings) 

        # get the scores for the most likely words, the linear layer, remove end word(?)
        linear_layer = self.hidden2vocab(lstm_out[:,:-1,:])   
        
        return linear_layer  


    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output_caption = []
        for i in range(max_len):
            
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.hidden2vocab(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            output_caption.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)     
   
        return output_caption