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



































from tensorflow.keras.datasets import mnist
import torch.nn as nn



# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Reshape images to flat vectors (28 * 28 = 784) and normalize to [0, 1]
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

print(train_images.shape)
print(test_images.shape)


# Convert data to torch tensors
train_data = torch.tensor(train_images.T, dtype=torch.float32)
train_targets = torch.tensor(train_labels.T, dtype=torch.long)  # CrossEntropyLoss expects long type

# Initialize the model and loss function
input_size = 28 * 28
output_size = 10  # 10 digits for MNIST
hidden_size = 800  # Can be set as per your model architecture

model = SimpleDFA(input_size, hidden_size, output_size)
model.Init_weights()  # Initialize weights

# Define the loss function (CrossEntropyLoss for multi-class classification)
loss_fn = nn.CrossEntropyLoss()  # for multi-class classification

# Train the network using DFA with MNIST dataset
train(model, train_data, train_targets, loss_fn, learning_rate=0.1, epochs=10000)

# Evaluate the model's predictions on the test set
with torch.no_grad():
    test_data = torch.tensor(test_images, dtype=torch.float32)
    test_outputs = model.forward(test_data)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == torch.tensor(test_labels, dtype=torch.long)).sum().item() / len(test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")