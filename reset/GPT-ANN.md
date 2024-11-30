Let's build a simple feedforward neural network with:

1. **Input layer**: 2 neurons (2 values in)
2. **Hidden layer**: 3 neurons
3. **Output layer**: 2 neurons

We will define example weight matrices and biases for each layer and show how matrix multiplication works for a forward pass through the network.

### Step 1: Define Inputs, Weights, and Biases

Let:
- **Input vector** \( \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \)

#### Weights and Biases for the Hidden Layer
- **Weight matrix for the hidden layer** \( \mathbf{W}_{\text{hidden}} \) has dimensions \( 3 \times 2 \) (3 neurons in the hidden layer, 2 input neurons):
  \[
  \mathbf{W}_{\text{hidden}} = \begin{bmatrix} 0.2 & 0.4 \\ -0.5 & 0.3 \\ 0.1 & -0.2 \end{bmatrix}
  \]

- **Bias vector for the hidden layer** \( \mathbf{b}_{\text{hidden}} \) has dimension \( 3 \times 1 \):
  \[
  \mathbf{b}_{\text{hidden}} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.05 \end{bmatrix}
  \]

#### Weights and Biases for the Output Layer
- **Weight matrix for the output layer** \( \mathbf{W}_{\text{output}} \) has dimensions \( 2 \times 3 \) (2 neurons in the output layer, 3 neurons in the hidden layer):
  \[
  \mathbf{W}_{\text{output}} = \begin{bmatrix} 0.3 & -0.6 & 0.8 \\ -0.1 & 0.4 & -0.5 \end{bmatrix}
  \]

- **Bias vector for the output layer** \( \mathbf{b}_{\text{output}} \) has dimension \( 2 \times 1 \):
  \[
  \mathbf{b}_{\text{output}} = \begin{bmatrix} 0.2 \\ -0.3 \end{bmatrix}
  \]

### Step 2: Calculate Hidden Layer Activations
The hidden layer activations \( \mathbf{h} \) can be calculated as:
\[
\mathbf{h} = \mathbf{W}_{\text{hidden}} \mathbf{x} + \mathbf{b}_{\text{hidden}}
\]

Let's perform the matrix multiplication:
1. Multiply \( \mathbf{W}_{\text{hidden}} \) by \( \mathbf{x} \):
   \[
   \mathbf{W}_{\text{hidden}} \mathbf{x} = \begin{bmatrix} 0.2 & 0.4 \\ -0.5 & 0.3 \\ 0.1 & -0.2 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} (0.2 \cdot 0.5) + (0.4 \cdot -0.5) \\ (-0.5 \cdot 0.5) + (0.3 \cdot -0.5) \\ (0.1 \cdot 0.5) + (-0.2 \cdot -0.5) \end{bmatrix} = \begin{bmatrix} -0.1 \\ -0.4 \\ 0.15 \end{bmatrix}
   \]

2. Add the bias \( \mathbf{b}_{\text{hidden}} \):
   \[
   \mathbf{h} = \begin{bmatrix} -0.1 \\ -0.4 \\ 0.15 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \\ 0.05 \end{bmatrix} = \begin{bmatrix} 0 \\ -0.6 \\ 0.2 \end{bmatrix}
   \]

For simplicity, let’s assume an identity activation function (no non-linearity), so the hidden layer output \( \mathbf{h} \) remains:
\[
\mathbf{h} = \begin{bmatrix} 0 \\ -0.6 \\ 0.2 \end{bmatrix}
\]

### Step 3: Calculate Output Layer Activations
The output layer activations \( \mathbf{y} \) can be calculated as:
\[
\mathbf{y} = \mathbf{W}_{\text{output}} \mathbf{h} + \mathbf{b}_{\text{output}}
\]

1. Multiply \( \mathbf{W}_{\text{output}} \) by \( \mathbf{h} \):
   \[
   \mathbf{W}_{\text{output}} \mathbf{h} = \begin{bmatrix} 0.3 & -0.6 & 0.8 \\ -0.1 & 0.4 & -0.5 \end{bmatrix} \begin{bmatrix} 0 \\ -0.6 \\ 0.2 \end{bmatrix} = \begin{bmatrix} (0.3 \cdot 0) + (-0.6 \cdot -0.6) + (0.8 \cdot 0.2) \\ (-0.1 \cdot 0) + (0.4 \cdot -0.6) + (-0.5 \cdot 0.2) \end{bmatrix} = \begin{bmatrix} 0.52 \\ -0.34 \end{bmatrix}
   \]

2. Add the bias \( \mathbf{b}_{\text{output}} \):
   \[
   \mathbf{y} = \begin{bmatrix} 0.52 \\ -0.34 \end{bmatrix} + \begin{bmatrix} 0.2 \\ -0.3 \end{bmatrix} = \begin{bmatrix} 0.72 \\ -0.64 \end{bmatrix}
   \]

### Final Output
The output of the network is:
\[
\mathbf{y} = \begin{bmatrix} 0.72 \\ -0.64 \end{bmatrix}
\]

This example illustrates the forward pass through a simple network with matrix multiplications and bias additions at each layer.
