import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    def __init__(self, layer_sizes, act_function, act_function_derivative, output_function=torch.sigmoid):
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
