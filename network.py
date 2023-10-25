import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

'''
My idea here is to try and build this 'hello world' neural network from scratch
using just numpy for some of the linear algebra stuff. i.e., I don't want to cheat 
with some logic already built into tensorflow.
'''

'''
One question: I'm not sure how to pick the number of hidden layers.
Searching around, it seems this is somewhat arbitrary.
Too many: might over fit, too few: might underfit.
Similarly, what should be the size of the hidden layer?
'''

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


# We know the MNIST input comes in as 28 x 28 pixel bitmaps of greyscale values
layer_sizes = [28*28, 30, 10]

num_layers = len(layer_sizes)

# Input layer doesn't need biases, as its values are not being changed, by definition.
biases = []
for nuerons_in_layer in layer_sizes[1:]:
    layer_biases_array = np.random.randn(nuerons_in_layer, 1)
    biases.append(layer_biases_array)

weights = []
for x, y in zip(layer_sizes[:-1], layer_sizes[1:]):
    # Give me an array of random numbers
    # of length y, where y is the number of neurons in the previous layer
    # and x is the number of neurons in the current layer

    # e.g., the first matrix of weights will have `30` rows
    # correspondign to each neuron in the second layer. Each row will have
    # 28*28 columns, corresponding to each neuron in the first layer.
    layer_weights = np.random.randn(y, x) # `y` x `x` matrix
    weights.append(layer_weights)

# Let's start training the network.
# Load MNIST dataset from Tensorflow
master_ds = tfds.load('mnist', split='train', shuffle_files=True)

# Shuffle all the data and fetch things in batches of 32
# Ref: the "stoachastic" in stoachastic gradient descent
batch_size = 1



test_ds, ds_info = tfds.load('mnist', split='test', shuffle_files=True, with_info=True)

# For each batch of images
# push one image through the network, which is to say
# set the activations of the first layer of neurons 
# to the values of the pixels in the image
for epoch in range(10):    
    batch_count = 0
    print(f"Epoch: {epoch}")
    ds = master_ds.shuffle(len(master_ds)).batch(batch_size)
    for batch_of_examples in ds:
        batch_count += 1
        

        image_batch = batch_of_examples['image']
        label_batch = batch_of_examples['label']

        # to start, there's no need to do anything.
        batch_bias_adjustments = [np.zeros(bias.shape) for bias in biases]
        batch_weight_adjustments = [np.zeros(weight.shape) for weight in weights]
        
        for i in range(batch_size):

            instance_bias_adjustments = [np.zeros(bias.shape) for bias in biases]
            instance_weight_adjustments = [np.zeros(weight.shape) for weight in weights]
        
            # Massage the input to more raw formats, for the sake of me doing this closer to the bare metal.
            # i.e., let's not rely on numpy or tensor flow to cover things up.

            # flatten 28x28x1 tensor into an array
            image = np.atleast_2d(image_batch[i].numpy().flatten()).transpose()
            image = image / 255.0 # normalize the values to be between 0 and 1
            
            # get the scalar int out of the tensor
            label_scalar = label_batch[i].numpy()
            label_array = np.zeros((10, 1))
            label_array[label_scalar] = 1.0

            # push this data through the existing network (i.e. with the current weights and biases)
            # The first layer is the image
            # where each neuron is the value of the pixel (already normalized to be between 0-1 above)
            layers_of_neurons = [image]
            biased_weighted_sums = []

            for bias, weight in zip(biases, weights):
                weighted_sum_of_previous_neurons = np.dot(weight, layers_of_neurons[-1])
                biased_weighted_sum_of_previous_neurons = weighted_sum_of_previous_neurons + bias
                biased_weighted_sums.append(biased_weighted_sum_of_previous_neurons)
                next_layer_of_neurons = sigmoid(biased_weighted_sum_of_previous_neurons)
                layers_of_neurons.append(next_layer_of_neurons)

            # So, to start, let's see what the cost (distance square from base) is for each image
            # 
            # (We don't actually need this thought)
            # result = layers_of_neurons[-1]
            # expected = np.zeros(10)
            # expected[label_scalar-1] = 1.0
            # cost = (result-expected)**2
            # print(cost)
            
            # Now we want to figure out how to adjust the weights and biases.

            # What change in weight/bias is going to have the biggest impact on the
            # average cost of all result neurons?
            
            # The derivative of the cost of a specific output neuron
            # will give us the slope of the direction we want to travel in.
            #
            # Expanding all of the above out:
            # cost(this_iteration) = (sigmoid(weights * layers_of_neurons[-1] + bias) - expected) ** 2
            #
            # To change the effect on _this_ neuron we can change its weights and bias or the state of the previous layer.
            #  
            # So, we're going to want to get the derivativs of the above cost with respect to A) weight and B) bias and C) the previous layer
            # and then use those slopes to "step" by some arbitrary amount in the direction of minimizing cost.
            # (...because finding the discrete minimum where dc/dw = 0 is too difficult given the number of parameters)
            # 
            # Calc refresher: https://www.youtube.com/watch?v=YG15m2VwSjA
            #                 https://www.justinmath.com/trick-to-apply-the-chain-rule-fast-peeling-the-onion/
            #
            # Working this out on paper you get:
            #
            # dc/dw = 2 * sigma ( weight * prev_activation + bias - expected ) * sigma_prime ( weight * prev_activation + bias ) * prev_activation
            #     
            # and
            #
            # dc/db = 2 * sigma ( weight * prev_activation + bias - expected ) * sigma_prime ( weight * prev_activation + bias )
            #
            # ...noting the only diff between the expresions is the activation value in the former.
            # 
            # Also we've already calculated the weight * activation + bias value in the previous step, so let's store it there.
            # 
            #
            # This is _only_ for the last layer of neurons, so we'll need to do this for each layer, working backwards.





            biased_weighted_sum = biased_weighted_sums[-1]
            dcdb = 2 * (layers_of_neurons[-1] - label_array) * sigmoid_prime(biased_weighted_sum)
            dcdw = np.dot(dcdb, layers_of_neurons[-2].transpose())
            instance_bias_adjustments[-1] = dcdb
            instance_weight_adjustments[-1] = dcdw
                    
        #     # For subsequent layers in the network, we'll need to take the derivative of the cost fuction with respect to
        #     # _that_ layer's weight and bias, which requires adding a few more chained terms, as we dig deeper into the network.
        #     # dc/dw = sigma_prime( weight * prev_activation + bias ) * prev_activation 
            for i in range(2, num_layers):
                biased_weighted_sum = biased_weighted_sums[-i]
                dcdb = np.dot(weights[-i+1].transpose(), dcdb) * sigmoid_prime(biased_weighted_sum)
                alt_dcdb = (np.sum(weights[-i+1].transpose()) * dcdb) * sigmoid_prime(biased_weighted_sum)
                instance_bias_adjustments[-i] = dcdb
                dcdw = np.dot(dcdb, layers_of_neurons[-i-1].transpose())
                instance_weight_adjustments[-i] = dcdw

            # Reduce the batch local adjustments into the global adjustments
            batch_bias_adjustments = [existing+delta for existing, delta in zip(batch_bias_adjustments, instance_bias_adjustments)]
            batch_weight_adjustments = [existing+delta for existing, delta in zip(batch_weight_adjustments, instance_weight_adjustments)]
            
        # We've now processed a batch, and we have some idea of the slope of the cost gradient
        # Let's move the weights and biases in that direction, by some arbitrary amount, proportional to
        # the size of the batch relative to the size of the training set.
        step_size = 1 / batch_size
        weights = [w-(step_size * nw) for w, nw in zip(weights, batch_weight_adjustments)]
        biases = [b-(step_size * nb) for b, nb in zip(biases, batch_bias_adjustments)]




    correct_count = 0
    i = 0
    for example in test_ds:
        i += 1
        # Massage the input to more raw formats, for the sake of me doing this closer to the bare metal.
        # i.e., let's not rely on numpy or tensor flow to cover things up.

        # flatten 28x28x1 tensor into an array
        image = np.atleast_2d(example['image'].numpy().flatten()).transpose()
        image = image / 255.0 # normalize the values to be between 0 and 1
        
        # get the scalar int out of the tensor
        label_scalar = example['label'].numpy()        

        # push this data through the existing network (i.e. with the current weights and biases)
        # The first layer is the image
        # where each neuron is the value of the pixel (already normalized to be between 0-1 above)
        layers_of_neurons = [image]
        biased_weighted_sums = []

        for bias, weight in zip(biases, weights):
            weighted_sum_of_previous_neurons = np.dot(weight, layers_of_neurons[-1])
            biased_weighted_sum_of_previous_neurons = weighted_sum_of_previous_neurons + bias
            biased_weighted_sums.append(biased_weighted_sum_of_previous_neurons)
            next_layer_of_neurons = sigmoid(biased_weighted_sum_of_previous_neurons)
            layers_of_neurons.append(next_layer_of_neurons)

        # So, to start, let's see what the cost (distance square from base) i
        expected = label_scalar
        actual = np.argmax(layers_of_neurons[-1])
        correct = expected == actual
        if correct:
            correct_count += 1
        
        print(f"Expected: {expected}, Actual: {actual}, Correct: {correct}, Correct Count: {correct_count}, total tries: {i}")
