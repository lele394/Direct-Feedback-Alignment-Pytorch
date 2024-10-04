# Free Project

## Day 1 26/09 - Test with a spiral - commit `e12a21c`

 - Spiral classification task : 2 spirals (here archimedean) are created. Network is trained to determine to which spiral the point belongs to. Then we try with a new test sets that implement noise, in the form of a random translation of the point (in a reasonnable limit).
 - Made a neural network using classic backpropagation
 - Tried multiple configuration, namely 3x50 hidden layers and using relu. 

### To Do Next

 - normalize output using softmax
 - use tanh activation function instead of relu.
 - try implementing DFA as a training algorithm for the network and replicate results.


## Day 2 03/10 - Implementating DFA - commit `----`

- Remade the implementation of using a class in test2.py
- Exposed the training loop and optimization algorithm
- Tried implementing specified additions above, but advised against
- Added a way to visualize the spacial classification using a grid of points

- For now : 3x50 and x2 output, all ReLU

### To Do Next
- Finish DFA implementation, it's not working yet.



























# Notes

What they call non linearity f seems to be the activation 

