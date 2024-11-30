# Free Project

## Day 1 26/09 - Test with a spiral - commit `e12a21c`

 - Spiral classification task : 2 spirals (here archimedean) are created. Network is trained to determine to which spiral the point belongs to. Then we try with a new test sets that implement noise, in the form of a random translation of the point (in a reasonnable limit).
 - Made a neural network using classic backpropagation
 - Tried multiple configuration, namely 3x50 hidden layers and using relu. 

### To Do Next

 - normalize output using softmax
 - use tanh activation function instead of relu.
 - try implementing DFA as a training algorithm for the network and replicate results.


## Day 2 03/10 - Implementating DFA - commit `8e10aaa`

- Remade the implementation of using a class in test2.py
- Exposed the training loop and optimization algorithm
- Tried implementing specified additions above, but advised against
- Added a way to visualize the spacial classification using a grid of points

- For now : 3x50 and x2 output, all ReLU

### To Do Next
- Finish DFA implementation, it's not working yet.


## Day 3 10/10 - Debug

- Debugging DFA
- Issue : Diverging a lot, goes to infinity, NaN results...

### To Do Next
- Make it work


## Day 4 17/10 - Keep debuggin

- Rewrote implementation
- Debugging DFA
- Issue : same as before (Diverging a lot, goes to infinity, NaN results...)

### To Do Next
- Make it work


## Day 5 24/10 - Studying instability

- Switched to XOR testing
- DFA is infact working but unstable (sometimes work, sometimes doesn't, check [the following](./a%20collection%20of%20broken%20training%20losses/info.md))
- Recording matrices for XOR and checking what makes it work.
- => Matrices conditions are irrelevant, system size was simply too small. "Statisticss is your friend" ~Daniel Brunner, 2024
- Starting work to support dynamic network sizes
### To Do Next
- switch to PyTorch
- Convert to a dynamic neural network


## Day 6 07/11 - Working prototype

- Switched to PyTorch using Anas' base work
- Finally successfully generalized the model to take an arbitrary number of hidden layers and size.
- Creating v1 library

### Remarks
- My implementation, though heavily based on Anas' doesn't seem to exhibit the diverging problems he faced for some reason.



## Day 7 14/11 - Working prototype, really

- Debugged issues with dimensions, was an error on the dimensions of e. Needed to do B_i * e for each row and do the average of each values.
- Fixed some dimensions issues. Code no longer strictly following Nokland's paper though. 

### Remarks
- Try to update bias and run on MNIST.


## Day 9 28/11 - Working prototype analog to Nokland's paper

- Finally have a prototype working with an implementation of DFA analog to Nokland's paper.
- Added `Sigmoid(X)` to the output.

- Successfully parametrized `forward_pass`.
- Successfully parametrized `dfa_backward_pass`.
- Successfully parametrized weights and bias updates.

- Added and reworked XOR problem solving.
- MNIST problem solved.

### Ongoing
- Parametrization of the number of layers and number of neurons. [DONE]
- Whole training comparison between DFA and Backprop graph. [TBD]
- Step comparison between DFA and Backprop graph. [TBD]

### Rework

- Moved `forward_pass` and `dfa_backward_pass` inside the `DynamicModel` class.
- Created `forward_pass_light` that acts as a basic forward pass that does not record activations. Please use it when `a` and `ħ` are not necessary.
- Parametrized the hidden layers, output, input, activation function and output activation function.

### Notes
- Check what's going on with feedback matrices dimensions. [DONE]














# Notes

What they call non linearity f seems to be the activation 

