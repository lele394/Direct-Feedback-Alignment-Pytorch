import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    def __init__(self, layer_sizes, act_function, act_function_derivative, output_function=torch.sigmoid):
        """
        Initialize the DynamicModel class.

        Args:
            layer_sizes (list[int]): Sizes of the layers, where each element represents the number of nodes in that layer.
            act_function (callable): Activation function to be applied between layers.
            act_function_derivative (callable): Derivative of the activation function, used for backpropagation.
            output_function (callable, optional): Output function to be applied to the final layer's output. Default is torch.sigmoid.

        Attributes:
            layers (nn.ModuleList): List of Linear layers constructed based on layer_sizes.
            layer_sizes (list[int]): Stores the provided layer sizes for reference.
            act_function (callable): The activation function for intermediate layers.
            act_function_derivative (callable): The derivative of the activation function.
            output_function (callable): The function applied to the output of the final layer.
        """
        super(DynamicModel, self).__init__()
        # Define layers using ModuleList
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        )
        self.layer_sizes = layer_sizes
        self.act_function = act_function  # Activation function
        self.act_function_derivative = act_function_derivative
        self.output_function = output_function

    # def forward(self, x):
    #     # Forward pass through each layer
    #     for i, layer in enumerate(self.layers):
    #         x = layer(x)
    #         # Apply activation to all except the last layer
    #         if i < len(self.layers) - 1:
    #             x = self.act_function(x)
    #     return x
    
    # Forward pass using PyTorch model
    def forward_pass_train(self, x):
        """
        This function performs the forward pass during training for a neural network.

        Parameters:
        - x: Input data for the network.

        Returns:
        - a: List of pre-activation values for each layer (excluding the input layer).
        - h: List of activations for each layer, including the input layer.
        - y_hat: The output of the network after applying the output function.

        Note: Removed the unnecessary 'x' at the start of the return statement. The function also assumes
        that `self.output_function` is defined and set elsewhere.
        """
        a = []
        h = [x]
        for l_index in range(len(self.layers)-1):
            # i-th layer
            a.append(self.layers[l_index](h[-1]))
            h.append(self.act_function(a[-1]))

        # last layer
        a_last = self.layers[-1](h[-1])
        y_hat = self.output_function(a_last)
        return a, h, y_hat #get rid of the x at the start

    # Forward pass getting rid of a and h list for simple use.
    def forward(self, x):
        """
        This function implements the forward pass of a neural network.

        Args:
            x: Input tensor.

        Returns:
            y_hat: The output of the neural network after passing through
                all layers and applying the activation and output functions.
        """
        a = 0
        h = x
        for l_index in range(len(self.layers)-1):
            # i-th layer
            a = self.layers[l_index](h)
            h = self.act_function(a)

        # last layer
        a_last = self.layers[-1](h)
        y_hat = self.output_function(a_last)
        return y_hat #get rid of the x at the start

    # DFA backward pass with PyTorch tensors
    def dfa_backward_pass(self, e, h, a, B): # x is h0 actually
        """
        Performs the Direct Feedback Alignment (DFA) backward pass.

        This function computes the weight (`dW`) and bias (`db`) gradients using 
        the error signal (`e`), hidden activations (`h`), pre-activation values (`a`), 
        and feedback weights (`B`). The gradients are calculated by applying the 
        activation function derivative to `a` and propagating the error signal through `B`.

        Args:
            e (torch.Tensor): Error signal tensor.
            h (list[torch.Tensor]): List of hidden activation tensors (including input as h[0]).
            a (list[torch.Tensor]): List of pre-activation tensors.
            B (list[torch.Tensor]): List of feedback weight tensors.

        Returns:
            tuple: 
                - dW (list[torch.Tensor]): List of weight gradients.
                - db (list[torch.Tensor]): List of bias gradients.
        """

        dW = []
        db = []

        for i in range(len(B)):
            da = torch.matmul(B[i], e) * self.act_function_derivative(a[i])
            dW.append(-torch.matmul(da, h[i].T))
            db.append(-torch.sum(da, dim=1, keepdim=True))

        dW.append(-torch.matmul(e, h[-1].T))
        db.append(-torch.sum(e, dim=1, keepdim=True))

        return dW, db
    


    def summary(self, input_shape):
        """
        This function generates a summary of the model architecture, including:
        - Layer index
        - Input and output shapes for each layer
        - Activation function type for each layer

        It performs a forward pass using a dummy input tensor of the specified input_shape 
        to determine the input and output shapes of each layer dynamically.

        The summary includes:
        - A header and footer for better readability
        - The input size, output size, and the sizes of all layers

        Parameters:
        - input_shape: A tuple representing the shape of the input tensor.

        This function assumes:
        - self.layers is an iterable containing the model layers.
        - self.act_function is the activation function applied after each layer (except the last layer).
        - self.output_function is the activation function applied after the last layer.
        - self.layer_sizes is a list containing the sizes of all layers.
        """
        print("===========================MODEL SUMMARY==============================")
        print(f"{'Layer':<10}{'Input Shape':<20}{'Output Shape':<20}{'Activation':<20}")
        print("=" * 70)
        
        x = torch.rand(*input_shape)  # Dummy input to trace the shapes
        for i, layer in enumerate(self.layers):
            input_shape = x.shape
            x = layer(x)  # Forward pass through the layer
            output_shape = x.shape
            activation = type(self.act_function) if i < len(self.layers) - 1 else type(self.output_function)
            activation = str(activation)
            print(f"{i:<10}{str(list(input_shape)):<20}{str(list(output_shape)):<20}{activation:<20}")
        print("=" * 70)
        print(f'input size  : {self.layer_sizes[0]}')
        print(f'output size : {self.layer_sizes[-1]}')
        print(f'layers sizes: {self.layer_sizes}')
        print("============================END SUMMARY===============================")
