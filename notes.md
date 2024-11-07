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























# Notes

What they call non linearity f seems to be the activation 

