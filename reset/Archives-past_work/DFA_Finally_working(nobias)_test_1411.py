import torch
import torch.nn as nn
import matplotlib.pyplot as plt








# Define the network architecture with a hidden layer and Tanh activation
class SimpleDFA(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        super(SimpleDFA, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input weight matrix (input to hidden)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # hidden weight matrix (hidden to hidden)
        self.fc3 = nn.Linear(hidden_size, output_size)  # hidden weight matrix (hidden to hidden)
        self.activation = nn.Tanh()  # Use Tanh activation function

    def forward(self, x):
        
        x = self.activation(self.fc1(x))  # Hidden layer with Tanh activation
        x = self.activation(self.fc2(x))  # Hidden layer with Tanh activation
        return self.activation(self.fc3(x))  # Output layer with Tanh activation
    
    
    def Init_weights(self):
        # Xavier init on weights
        nn.init.xavier_uniform_(self.fc1.weight) 
        nn.init.xavier_uniform_(self.fc2.weight) 
        nn.init.xavier_uniform_(self.fc3.weight) 
        
    
    def Init_feedback(self):
        
        self.B_1 = torch.randn(self.input_size, self.output_size) * 0.01  # Feedback for 1st layer
        # self.B_1 = torch.randn(self.output_size, self.output_size) * 0.01  # Feedback for 1st layer
        self.B_2 = torch.randn(self.hidden_size, self.output_size) * 0.01 # Feedback for 2nd layer
        
        # self.B_1 = torch.randn( 6 , 5 ) * 0.01  # Feedback for 1st layer
        # self.B_2 = torch.randn( 2 , 5 ) * 0.01 # Feedback for 2nd layer  # NOT YET

        #normalize the feedback matrices to avoid exploding feedback
        self.B_1 = self.B_1 / self.B_1.norm()
        self.B_2 = self.B_2 / self.B_2.norm()
        return self.B_1, self.B_2
    
    def forward_pass(self,Input,Target,loss_func):
        
        # Forward pass
        Output = self.forward(Input)
        loss = loss_func(Output, Target)
        return Output, loss
        
    def activation_derivative(self, x):
        
        return 1 - self.activation(x)**2  # Derivative of Tanh activation function
    



























































def dfa_found_update_step(model, B2, B1, output, target, a1, a2, h1, h2, inputs, learning_rate=0.01):

    # This has been written to follow the article



    B1 = model.B_1
    B2 = model.B_2


    # Compute error at the output layer
    e = output - target  # Error 
    
    # DFA specific
    # print("e :\n", e)
    # print("B2 :\n", B2)
    # print("B1 :\n", B1)
    # print("a1 :\n", a1)
    # print("a2 :\n", a2)
    # print("h1 :\n", h1)
    # print("h2 :\n", h2)

    # print(model.activation_derivative(a2) )
    # print(torch.matmul(B1, e[0].T))

    # Removing batch dimension by averaging
    e_1 = (torch.matmul(B1, e.T).sum(dim=0) / e.shape[0]).unsqueeze(0)
    e_2 = (torch.matmul(B2, e.T).sum(dim=0) / e.shape[0]).unsqueeze(0)

    # print(e_1)
    


    # da1 = torch.matmul(B1, e.T) * model.activation_derivative(a1)  
    da1 = e_1.T * model.activation_derivative(a1)  
    # print("da1 :\n", da1)
    # da2 = torch.matmul(B2, e.T) * model.activation_derivative(a2)  
    da2 = e_2.T * model.activation_derivative(a2)  
    # print("da2 :\n", da2)


    # Common Updates
    dW1 = -torch.matmul(da1.T, inputs) 
    # db1 = -torch.sum(da1, dim=1, keepdim=True) 
    
    dW2 = -torch.matmul(da2.T, h1) 
    # db2 = -torch.sum(da2, dim=1, keepdim=True) 

    dW3 = -torch.matmul(e.T, h2) 
    # db3 = -torch.sum(e, dim=1, keepdim=True) 


    # Update the model's weights and biases
    with torch.no_grad():
        model.fc1.weight += learning_rate * dW1 
        # model.fc1.bias -= learning_rate * db1.squeeze()  
        
        model.fc2.weight += learning_rate * dW2  
        # model.fc2.bias -= learning_rate * db2.squeeze()  

        model.fc3.weight += learning_rate * dW3  
        # model.fc3.bias -= learning_rate * db3.squeeze()  


































def train(model, data, targets, loss_fn, learning_rate=0.01, epochs=1000):
    # Initialize feedback matrices
    B_1, B_2 = model.Init_feedback()

    losses = []  # To store the loss at each epoch
    for epoch in range(epochs):
        # Forward pass
        a1 = model.fc1(data)  # Get hidden layer activations
        h1 = model.activation(a1)

        a2 = model.fc2(h1)  # Get hidden layer activations
        h2 = model.activation(a2)

        ay = model.fc3(h2)
        y_hat = model.activation(ay)




        output, loss = model.forward_pass(data, targets, loss_fn)

        # Apply DFA update step
        # dfa_update_step(model, B_1, B_2, output, targets, hidden_activations, data, learning_rate)
        dfa_found_update_step(model, B_1, B_2, output, targets, a1, a2, h1, h2, data, learning_rate)

        # Store the loss
        losses.append(loss.item())

        # Print loss and accuracy every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")

            # Compute accuracy
            predictions = (output >= 0.5).float()  # Threshold the output to get binary predictions
            correct = (predictions == targets).float().sum()
            accuracy = correct / targets.size(0)
            print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Plot the convergence of loss
    plt.plot(losses)
    plt.title("Convergence of Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()







































# Example usage with XOR data
input_size = 2  # XOR has two input features (0 or 1)
hidden_size = 50  # You can adjust this value for the hidden layer
output_size = 1  # XOR has one output feature (0 or 1)



# data = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.float32)  # test 5 inputs
# targets = torch.tensor([[0, 0], [1, 0], [1, 0], [0, 0], [0, 0]], dtype=torch.float32)  # test 5 targets

# data = torch.tensor([[0, 0, 0]], dtype=torch.float32)  # test 5 inputs
# targets = torch.tensor([[0, 0]], dtype=torch.float32)  # test 5 targets


# # Define inputs for Full Adder (A, B, C_in)
# data = torch.tensor([
#     [0, 0, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 1, 1],
#     [1, 0, 0],
#     [1, 0, 1],
#     [1, 1, 0],
#     [1, 1, 1]
# ], dtype=torch.float32)

# # Define outputs for Full Adder (Sum, C_out)
# targets = torch.tensor([
#     [0, 0],  # A=0, B=0, C_in=0 -> Sum=0, C_out=0
#     [1, 0],  # A=0, B=0, C_in=1 -> Sum=1, C_out=0
#     [1, 0],  # A=0, B=1, C_in=0 -> Sum=1, C_out=0
#     [0, 1],  # A=0, B=1, C_in=1 -> Sum=0, C_out=1
#     [1, 0],  # A=1, B=0, C_in=0 -> Sum=1, C_out=0
#     [0, 1],  # A=1, B=0, C_in=1 -> Sum=0, C_out=1
#     [0, 1],  # A=1, B=1, C_in=0 -> Sum=0, C_out=1
#     [1, 1]   # A=1, B=1, C_in=1 -> Sum=1, C_out=1
# ], dtype=torch.float32)

data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # XOR inputs
targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR targets

# Initialize the model and loss function
model = SimpleDFA(input_size, hidden_size, output_size)
model.Init_weights()  # Initialize the weights

loss_fn = nn.MSELoss()  

# Train the network using DFA and print XOR accuracy
train(model, data, targets, loss_fn, learning_rate=0.1, epochs=10000)

print(model.forward(data))