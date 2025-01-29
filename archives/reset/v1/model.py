import torch
import torch.nn as nn


class SimpleDFA(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, hidden_layers_number=1):
        
        super(SimpleDFA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers_number = hidden_layers_number
        self.output_size = output_size
        
        # Use ModuleList for layers
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_size)] + 
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers_number)] + 
            [nn.Linear(hidden_size, output_size)]
        )
        
        self.activation = nn.Tanh()  # Use Tanh activation function

    def forward(self, x):
        # Apply activation to each hidden layer
        for layer in self.layers[:-1]:  # All except the last layer
            x = self.activation(layer(x))
        # Final layer (output layer) without activation
        x = self.layers[-1](x)
        return x
    
    
    def Init_weights(self):
        # Xavier init on weights
        for layer in self.layers: 
            nn.init.xavier_uniform_(layer.weight) 

    
    def Init_feedback(self):
        # Create feedback matrices matching each layer's output size
        self.B_hiddens = [torch.randn(self.hidden_size, self.output_size) * 0.01 for _ in range(self.hidden_layers_number)]
        self.B_output = torch.randn(self.output_size, self.output_size) * 0.01  # Feedback for output layer

        # Normalize feedback matrices
        for i in range(len(self.B_hiddens)):
            self.B_hiddens[i] /= self.B_hiddens[i].norm()
        self.B_output /= self.B_output.norm()

        return self.B_hiddens + [self.B_output]  # Feedback matrices for hidden layers + output


    def forward_pass(self,Input,Target,loss_func):
        
        # Forward pass
        Output = self.forward(Input)
        loss = loss_func(Output, Target)
        return Output, loss
        
    def activation_derivative(self, x):
        
        return 1 - self.activation(x)**2  # Derivative of Tanh activation function
    