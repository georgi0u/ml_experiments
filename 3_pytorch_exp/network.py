import os
import random
import signal
import sys

import numpy as np
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab, hidden_size):
        super(RNN, self).__init__()

        self.vocab = vocab

        self.char_to_index = {c: i for i, c in enumerate(vocab)}
        self.index_to_char = {i: c for i, c in enumerate(vocab)}

         
        self.input_size = len(vocab)
        self.hidden_size = hidden_size

        self.cell = nn.LSTM(self.input_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(in_features=hidden_size, out_features=self.input_size)

        self.eye = torch.eye(self.input_size, requires_grad=False)
        

    def forward(self, input_char, h, c):
        input_tensor = self.one_hot(input_char)
        output, (hidden, cell) = self.cell(input_tensor, (h,c))        
        output = self.dropout(output)
        output = self.output(output)
        return  output, (hidden, cell)
    
    def one_hot(self, character):
        index = self.char_to_index[character]
        if index == None:
            raise "Character not in vocab"
        if index < 0 or index >= self.input_size:
            raise "Character is out of bounds. Did you init things correctly?"
        
        # if you don't make a copy, running this on the mps device fails? 
        return torch.clone(self.eye[index]).unsqueeze(1).T

def main(data):
    device = torch.device('cpu')
    torch.set_default_device(device)

    sample_size = 30
    hidden_size = 128
    vocab = list(set(data))


    rnn = RNN(vocab, hidden_size).to(device=device)    
    if os.path.isfile('model_weights.pth'):
        rnn.load_state_dict(torch.load('model_weights.pth'))

    def sig_handle(sig, frame):
        torch.save(rnn.state_dict(), 'model_weights.pth')
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handle)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=.002, momentum=.09)

    position = 0
    while position + sample_size + 1 < len(data):
        rnn.train()
        optimizer.zero_grad()

        input_sample = data[position : position+sample_size]
        expected_output_sample = data[position+1 : position+sample_size+1]
        
        hidden, cell = torch.zeros(1, hidden_size), torch.zeros(1, hidden_size)
        loss = torch.zeros(1, device=device)
        for i, _ in enumerate(input_sample):
            input_char = input_sample[i]
            expected_output_char = expected_output_sample[i]
            expected_output_char_vector = rnn.one_hot(expected_output_char)

            output, (hidden, cell) = rnn(input_char, hidden, cell)
            
            loss_local = loss_fn(output, expected_output_char_vector)
            
            loss += loss_local
        
        loss.backward()
        optimizer.step()
        
        if position % 100000 == 0:
            with torch.no_grad():  # no need to track history in sampling
                start_char = random.sample(rnn.vocab, 1)[0]
                input = start_char
                
                hidden, cell = torch.zeros(1, rnn.hidden_size), torch.zeros(1, rnn.hidden_size)
                output_str = ''
                for i in range(200):
                    output, (hidden, cell) = rnn(input, hidden, cell)
                    distro = output.cpu().detach().softmax(dim=1).numpy().ravel()
                    output_i = np.random.choice(range(rnn.input_size), p=distro)
                    output_char = rnn.index_to_char[output_i]
                    output_str += output_char
                    input = output_char
                
                print(f"Position {position}, Loss {loss.item()}")
                print('<start>')
                print(output_str)
                print('</end>')
                print('\n\n\n')
        position += sample_size   
    

if __name__ == '__main__':
    input_data_filename = sys.argv[1]
    with open(input_data_filename) as fp:
        data = fp.read()
    main(data)