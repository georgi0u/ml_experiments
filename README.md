# Experimenting with basic neural networks.

## Starting Points:

* https://www.3blue1brown.com/topics/linear-algebra
* http://neuralnetworksanddeeplearning.com/
* https://www.3blue1brown.com/topics/neural-networks

## Checkpoint 1

Implement basic neural net to classify  MNIST database of hand drawn digit characters.

Start in python: relatively easy, but can get a bit script-kid-y-ish relying on numpy's APIs to obscure some of the matrix math.

Reproduce in C++ using the Eigen library which requires one to be a bit more verbose. e.g., `numpy.dot`
seems to be implemented to do whatever is most context-likely. Eigen forces you to say you want a dot product,
or a element product, etc. Also it's fun to see the training get fast, without yet touching a GPU.

## Checkpoint 2

Investigate a bit of cross entropy cost fucntion

## Checkpoint 3

Stop coding a bit and skim the rest of http://neuralnetworksanddeeplearning.com/.

Learn the concepts of, if not the details of:

* Vanishing/Exploding Gradient Problem:
  * The gradient in early layers is the product of terms from all the later layers. If those later layer's definitions produce small outputs, the early layers will be smaller still. Or visa versa. Eventually, your "steps" either stop being signification or way overshoot the shape of the gradient.
* Regularization:
    * Prefer small weights so we don't get so specific of an inferred polynomial.
    * Reduces overfitting.
    * Accomplished by, e.g., adding an extra term to the cost function like the sum of the squares of *all* weights.
        * Cost is low when weights are low OR when a large weight reduces the precceeding term significatly enough to outweight
	  the new second term.
* Weight Initialzation:
  * tl;dr: you can do better than random by trying to reduce large weights to begin with.
* Softmax:
  * Don't use one of the standard/simpler non-linear (sigmoid, relu) functions to normalize output.
  * Instead define activation of one neuron as a fraction  of all other neurons, effecitvely turning the output layer into a probability distribution.

## Checkpoint 4

* Watch [lecture on Convolutions](https://www.youtube.com/watch?v=KuXjwB4LzSA)

Convolutions can let us extract "features" from an input.
e.g., w/r/t imagery, we can get edges.

Map these convolutions of the input to a hidden layer using a single set of "shared" weights and biases and you get a feature map.

### Note

This is probably worth going back and implementing.


## Checkpoint 5

Attempt some more complex papers:

* [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)


## Checkpoint 6

_Go back and learn more about RNNs._

* https://karpathy.github.io/2015/05/21/rnn-effectiveness/
* https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/


