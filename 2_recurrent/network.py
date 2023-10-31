import numpy as np
import torch
import torch.nn as nn


# Just starting to learn the Torch API.
# It seems to be able to do a lot of the work for me, but I only want to use it here for:
#   A) Some simple organization and
#   B) Routing the network to the GPU
# Otherwise, I'd like to do this as close to the metal as possible, for learning purposes.
class RNN():
    def __init__(self, data, hidden_size, sample_size):
        """
        This will be a small-ish data set, so let's take in all the data and figure out 
        the vocab embedding from it.
        """
        super().__init__()

        # We'll need to convert the input character to vectors and back again.
        
        chars = list(set(data))
        self.num_chars = len(chars)

        self.char_to_index = { char:i for i,char in enumerate(chars) }
        self.index_to_chars = { i:char for i,char in enumerate(chars) }
        self.one_hots = torch.eye(self.num_chars)

        self.num_hidden_neurons = hidden_size

        self.sample_size = sample_size
        
        # Note:
        # PyTorch "parameters" are "registered" with the "Module"
        # and allow for users to affect parameters and their calculations
        # in mass. e.g., push the calculations to the gpu, or in my case, on 
        # this M2 mac, to the gpu-ish thing.

        # ...Well scratch that.

        # Weights
        self.weights_input_to_hidden = (
            torch.randn(self.num_hidden_neurons, self.num_chars) * .01)
        self.weights_hidden_to_next_hidden = (
            torch.randn(self.num_hidden_neurons, self.num_hidden_neurons) * .01)
        self.weights_hidden_to_output = (
            torch.randn(self.num_chars, self.num_hidden_neurons) * .01)
        
        # Biases
        self.hidden_layer_biases = (
            torch.zeros(self.num_hidden_neurons, 1))
        self.output_layer_biases = (
            torch.zeros(self.num_chars, 1))
        
        self.hidden_states = [torch.zeros(self.num_hidden_neurons, 1) for i in range(self.sample_size)]

    def reset_mem(self):
        self.hidden_states = [torch.zeros(self.num_hidden_neurons, 1) for i in range(self.sample_size)]

    def one_hot(self, character):
        index = self.char_to_index[character]
        if index == None:
            raise "Character not in vocab"
        if index < 0 or index >= self.num_chars:
            raise "Character is out of bounds. Did you init things correctly?"
        
        # if you don't make a copy, running this on the mps device fails? 
        return torch.clone(self.one_hots[index]).unsqueeze(1)
        
    def forward(self, input_sample, targets):
        if (len(input_sample) != self.sample_size):
            raise f"input_sample wrong size {len(input_sample)}"
        
        
        outputs = []
        
        loss = 0
        for i, character in enumerate(input_sample):
            input_vector = self.one_hot(character)
            
            weighted_input_vector = self.weights_input_to_hidden @ input_vector

            weighted_hidden_state =  self.weights_hidden_to_next_hidden @ self.hidden_states[i-1]
            smushed_value = torch.tanh(weighted_input_vector + weighted_hidden_state + self.hidden_layer_biases)
            self.hidden_states[i] = smushed_value
            
            pre_biased_output = self.weights_hidden_to_output @ smushed_value

            # Output (singular) will be a vector of values representing guesses at
            # which character comes after the input character.
            output = pre_biased_output + self.output_layer_biases
            
            softmax_output = torch.exp(output) / torch.sum(torch.exp(output))

            # Outputs (plural) will be a list of character vectors
            outputs.append(softmax_output)
            
            target_index = self.char_to_index[targets[i]]

            loss += -torch.log(softmax_output[target_index,0])

        return (outputs, loss.item())

    def backward(self, sampled_input, outputs, targets):
        """
        Initial reading of the PyTorch docs shows there's some magic that lets you "auto-diff" as in
        auto-differentiate your forward pass and descend automatically. But, since we're trying to
        learn here, let's do it by hand.
        """
        delta_weights_input_to_hidden = torch.zeros_like(self.weights_input_to_hidden)
        delta_weights_hidden_to_next_hidden = torch.zeros_like(self.weights_hidden_to_next_hidden)
        delta_weights_hidden_to_output = torch.zeros_like(self.weights_hidden_to_output)

        delta_hidden_layer_biases = torch.zeros_like(self.hidden_layer_biases)
        delta_output_layer_biases = torch.zeros_like(self.output_layer_biases)

        delta_hidden_next = torch.zeros(self.num_hidden_neurons, 1)

        n = 0
        
        
        for i in reversed(range(len(sampled_input))):
            n += 1
            d_output = outputs[i].clone()            
            d_output[self.char_to_index[targets[i]]] -= 1

            input_vector = self.one_hot(sampled_input[i])
            
            delta_weights_hidden_to_output += d_output @ self.hidden_states[i].t()

            delta_output_layer_biases += d_output

            delta_hidden_state = (self.weights_hidden_to_output.t() @ d_output) + delta_hidden_next
            
            delta_hidden_raw = (1 - (self.hidden_states[i] * self.hidden_states[i])) * delta_hidden_state
            delta_hidden_layer_biases += delta_hidden_raw
            delta_weights_input_to_hidden += delta_hidden_raw @ input_vector.t()

            delta_weights_hidden_to_next_hidden += delta_hidden_raw @ self.hidden_states[i-1].t()
            
            delta_hidden_next = self.weights_hidden_to_next_hidden.t() @ delta_hidden_raw

            

        deltas = [
             delta_weights_input_to_hidden,
             delta_weights_hidden_to_next_hidden,
             delta_weights_hidden_to_output,
             
             delta_hidden_layer_biases,
             delta_output_layer_biases,
        ]

        for delta in deltas:
            torch.clip(delta, -5, 5, out=delta)
            
        return deltas
    

    def sample(self, seed_char, range_):
        input_vector = self.one_hot(seed_char)
        output_str = ''
        prev_hidden_state = self.hidden_states[-1]
        for _ in range(range_):
            weighted_input_vector = self.weights_input_to_hidden @ input_vector
            weighted_hidden_state =  self.weights_hidden_to_next_hidden @ prev_hidden_state

            smushed_value = torch.tanh(weighted_input_vector + weighted_hidden_state + self.hidden_layer_biases)
            prev_hidden_state = smushed_value
            
            pre_biased_output = torch.matmul(self.weights_hidden_to_output, smushed_value)

            # Output (singular) will be a vector of values representing guesses at
            # which character comes after the input character.
            output = pre_biased_output + self.output_layer_biases
            output_exp = torch.exp(output)
            softmax_output = output_exp / torch.sum(output_exp)

            # Outputs (plural) will be a list of character vectors
            #idx = torch.argmax(softmax_output).item()
            idx = np.random.choice(range(self.num_chars), p=softmax_output.numpy().ravel())
            
            output_char = self.index_to_chars[idx]
            
            output_str += output_char
            input_vector = self.one_hot(output_char)

        return output_str
    



with open('input.txt', 'r') as f:
    data = f.read()



#device = torch.device("mps")
#torch.set_default_device(device)
sample_size = 30
torch.set_printoptions(precision=8)
rnn = RNN(data, hidden_size=512, sample_size=sample_size)

position = 0
i = 0

mWxh, mWhh, mWhy = torch.zeros_like(rnn.weights_input_to_hidden), torch.zeros_like(rnn.weights_hidden_to_next_hidden), torch.zeros_like(rnn.weights_hidden_to_output)
mbh, mby = torch.zeros_like(rnn.hidden_layer_biases), torch.zeros_like(rnn.output_layer_biases) # memory variables for Adagrad

while True:
    if (position + sample_size) >= len(data) or i == 0:
        rnn.reset_mem()
        position = 0
        
    sample = data[position:position+sample_size]
    targets = data[position+1:position+sample_size+1]
    outputs, loss = rnn.forward(sample, targets)

    if i % 1000 == 0:

        print(f'Iteration {i}: loss: {loss}' )
        print(rnn.sample(data[position], 500))
        print('</end>')

    deltas = rnn.backward(sample, outputs, targets)
    
    for param, delta, mem in zip([rnn.weights_input_to_hidden, rnn.weights_hidden_to_next_hidden, rnn.weights_hidden_to_output, rnn.hidden_layer_biases, rnn.output_layer_biases],                                 
                                 deltas,
                                 [mWxh, mWhh, mWhy, mbh, mby]):

        mem += delta * delta
        param += -.1 * delta / torch.sqrt(mem + 1e-8) # adagrad update

    position = position + sample_size
    i += 1
    