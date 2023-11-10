# Experimenting with basic neural networks.

## Starting Points:

* https://www.3blue1brown.com/topics/linear-algebra
* http://neuralnetworksanddeeplearning.com/
* https://www.3blue1brown.com/topics/neural-networks

## Checkpoint 1

Implement basic neural net to classify MNIST database of hand drawn digit characters.

Start in python: relatively easy, but can get a bit script-kid-y-ish relying on numpy's APIs to obscure some of the matrix math.

Reproduce in C++ using the Eigen library which requires one to be a bit more verbose. e.g., `numpy.dot`
seems to be implemented to do whatever is most context-likely. Eigen forces you to say you want a dot product,
or a element product, etc. Also it's fun to see the training get fast, without yet touching a GPU.

## Checkpoint 2

Investigate a bit of cross entropy cost function

## Checkpoint 3

Stop coding a bit and skim the rest of http://neuralnetworksanddeeplearning.com/.

Learn the concepts of, if not the details of:

* Vanishing/Exploding Gradient Problem:
  * The gradient in early layers is the product of terms from all the later layers. If those later layer's definitions produce small outputs, the early layers will be smaller still. Or visa versa. Eventually, your "steps" either stop being signification or way overshoot the shape of the gradient.
* Regularization:
    * Prefer small weights so we don't get so specific of an inferred polynomial.
    * Reduces over-fitting.
    * Accomplished by, e.g., adding an extra term to the cost function like the sum of the squares of *all* weights.
        * Cost is low when weights are low OR when a large weight reduces the preceding term significantly enough to outweigh
	  the new second term.
* Weight Initialization:
  * tl;dr: you can do better than random by trying to reduce large weights to begin with.
* Softmax:
  * Don't use one of the standard/simpler non-linear (sigmoid, relu) functions to normalize output.
  * Instead define activation of one neuron as a fraction  of all other neurons, effectively turning the output layer into a probability distribution.

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
* https://towardsdatascience.com/recurrent-neural-networks-and-natural-language-processing-73af640c2aa1
* https://medium.com/@mliuzzolino/hello-rnn-55a9237b7112
* https://www.youtube.com/watch?v=SEnXr6v2ifU

Ok, implemented a basic RNN.

The main idea of persisting state through "cells" of networks, allowing you to keep track of sequence.

Read a bit a bout GRU and LSTM: mechanisms for keeping track of which parts of previous state make sense to pay attention to and which don't.

You can see my vanilla implementation struggle for lack of a sequential _relevance_ (i.e. lstm) mechanism.

This becomes relevant, a lá https://www.youtube.com/watch?v=SEnXr6v2ifU, when you start to think about RNN encoders and decoders — first mention of a formal Attention mechanism (precursor to transformers).

Also began to play with PyTorch, in order to to try and move stuff the Mac's GPU. Funny enough, things run slower there.

And.. PyTorch is a python wrapper of Torch, which has it's own kind of gross language for generally specifying gpu algos.

The thing I learned from exploring py torch and playing around with this:

Most of my experimentation has been with trying to implement things as close to the math as possible.

...and the place I get mostly tripped up at is: trying to do the gradient derivatives by hand on paper, and then translate them to code.

I spend about 10% of the time understanding a new concept (e.g. softmax, cross-entropy, clipping, etc) and then 90% of the time ensuring my matrixes are the correct shape and my derivatives are the right form.

e.g. I wasted a whole day accidentally multiplying something by a delta_weight matrix instead of the original weight matrix

And anyway, as I've learned about PyTorch, I'm seeing they have mechanisms for automatically computing the derivatives of various tensors you might want to compute gradients for, so long as you construct those tensors and manipulate them using PyTorch in the first place (i.e. don't sneak a numpy matrix in there.)

And and and! I'm thinking to myself: how much of "being productive" here is going to be (A) learning all the deep fundamentals and then trying to be something of a research scientist/academic, improving Machine Learning as a field, generally; vs (B) using the fundamentals as a foundation for leveraging higher level tools to build useful stuff?

Where's that line?

## Checkpoint 7

More reading:

* https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/
* https://gwern.net/scaling-hypothesis

To-Do:
* Practice some napkin math to calculate GPU requirements from Parameter size.
* Let's _not_ do our own derivatives, and see how far we can get with PyTorch
* Re-vist some of the earlier papers I didn't understand before I learned RNNs.

## Checkpoint 8

* Let's backup, and read through https://pytorch.org/tutorials/
  * This is so fucking cool. Ha.
  * Everything up to their ["Build the Neural Network"](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) chapter was inferable enough, as seen above.
  * But they've gone ahead and abstracted all the layering, differentiation, and optimization out!

Ok, got an LSTM based RNN working, on the shakespear stuff. Played around with some of the new hyperparameters made easy by pytorch.
* Picked my own learning rate, with some experimentation
* Learned a bit about SGD with momentum.

Also read a bit about generalization: [Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models]
(https://arxiv.org/abs/2311.00871)

## Checkpoint 8

Went back to [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and understood a lot more of it. I have a basic idea of attention, but I'm still not completely understanding the difference between a traditional RNN with attention and an attention-only transformer.

Let's give this pytroch tutorial a shot: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html.

## Checkpoint 9

Switched gears, back to the day job.

Spent some time looking into production-izing of open models.

Checked out llama-2, directly from meta. Hit the (apparently typical) roadblocks when trying to run on the Mac laptop: 
* Meta's code is distributed as a "raw" PyTorch model (pth), and their provided instantiating code is — at first glance — hardwired to use Nvidia/cuda hardware. Not sure if a `torch.set_default_device` would let that work out of the box, but didn't feel like going down that route.
* Secondly, even the "small" 7B param model is ~20 Gigs big. Not fitting on my 32gig laptop without swapping.

Moved on to find llama-cpp and ggerganov's quantized versions of llama-2. Curious point that removing weight/bias precision does *not* kill performance/accuracy as much as you might expect.

So, got that working well enough. Is **that** worth production-izing?

Looked into some go bindings and LocalAI.
* https://github.com/go-skynet/go-llama.cpp
* https://github.com/mudler/LocalAI

Seems promising, but...

Do I really want to be relying on a privately maintained project? How robust is this stuff?

My intuition is: one can use the weights, but the projects are still a bit too early and also heavy for my use. As in, I don't want everything LocalAI has to offer. And I don't want to deal with ggerganov's formats until they're more robust/time tested/persistent.

^ An opinion somewhat biased by my parallel discovery of everything going on at HuggingFace.

HuggingFace has their own formats, with convenient ways for fine tunning and saving "checkpoint" files describing changes to underlying base models!

Great! 

Further, their [`transformers`](https://github.com/huggingface/transformers) library seems much more application friendly than trying to wrap up PyTorch in an application layer. 

e.g., it's more declarative and makes reasonable assumptions to minimize initial configuration. 

So for fun, got one of ggerganov's quantized models converted to a huggingface format, and got that running through the transformers lib.

Moving on, I also found Anyscale.com and https://ray.io/ <- seems this is the converging on canonical way to productionize _any_ model, allowing for the segregation of serving ingress servers and models, as well as the managed-allocation of resources for each (GPUS, CPUs across replicas, etc.). And this all works with k8. Great.

Started toying around with that, allocating a couple NVIDIA cards on GCP, but ran into headaches with drivers. Got far enough to prove the point though.