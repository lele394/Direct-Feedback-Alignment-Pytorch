import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import numpy as np
from scipy.special import expit

"""
Notes to future me or other users.

The following does not support the use of custom activation functions for each layer (low-key want to add that though. I might).

"""



class DynamicModel(nn.Module):
    def __init__(self, layer_sizes, act_function, act_function_derivative, output_function=torch.sigmoid):
        super(DynamicModel, self).__init__()
        # Define layers using ModuleList
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        )
        self.act_function = act_function  # Activation function
        self.act_function_derivative = act_function_derivative
        self.output_function = output_function

    def forward(self, x):
        # Forward pass through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all except the last layer
            if i < len(self.layers) - 1:
                x = self.act_function(x)
        return x
    
    # Forward pass using PyTorch model
    def forward_pass(self, x):

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
    def forward_pass_light(self, x):

        a = 0
        h = x
        for l_index in range(len(self.layers)-1):
            # i-th layer
            a = self.layers[l_index](h)
            h = self.act_function(a)

        # last layer
        a_last = self.layers[-1](h)
        y_hat = torch.sigmoid(a_last)
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


# ====================================================================================================================================


def average_angle(W2, B1, error, a1, a2):
    dh1 = np.mean(np.matmul(B1, error), axis=1, keepdims=True)  # Maybe no derivative needed here
    c1 = np.mean(np.matmul(np.transpose(W2), error * (expit(a2) * (1 - expit(a2)))), axis=1, keepdims=True)
    dh1_norm = np.linalg.norm(dh1)
    c1_norm = np.linalg.norm(c1)
    inverse_dh1_norm = np.power(dh1_norm, -1)
    inverse_c1_norm = np.power(c1_norm, -1)
    
    # ALIGNMENT CRITERION AND ANGLE
    Lk = (np.matmul(np.transpose(dh1), c1) * inverse_dh1_norm)[0, 0]
    beta = np.arccos(np.clip(Lk * inverse_c1_norm, -1., 1.)) * 180 / np.pi
    return Lk, beta



# Training loop
def train(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize random feedback matrix
    B = [torch.randn(layer.out_features, model.layers[-1].out_features) for layer in model.layers[:-1]]

    # Dataset conversion stuff
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Recorded metrics
    angles = []
    te_dfa = []
    loss_dfa = []

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        train_error = 0

        # Batch run
        for batch_x, batch_y in dataloader:
            # Forward pass
            a, h, y_hat = model.forward_pass(batch_x)

            # Error and error metric
            error = y_hat - batch_y
            preds = torch.argmax(y_hat, dim=1)
            truth = torch.argmax(batch_y, dim=1)
            train_error += (preds != truth).sum().item()

            # Loss metric
            loss_on_batch = F.binary_cross_entropy(y_hat, batch_y)
            epoch_loss += loss_on_batch.item()

            # Transposition of a and h for dimensions match
            a = [matrix.T for matrix in a]
            h = [matrix.T for matrix in h]

            # DFA backward pass
            dW, db= model.dfa_backward_pass(error.T, h, a, B)
            


            # Update weights manually
            with torch.no_grad():
                for i in range(len(db)):
                    model.layers[i].weight += lr * dW[i]
                    model.layers[i].bias += lr * db[i].squeeze()

            # Not implemented yet
            # if len(angles) % 100 == 0:
                # angles.append(average_angle(model.layers[1].weight.T, B1, error.T, a1.T, a2.T))

        training_error = train_error / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}, Training Error = {training_error:.4f}")

        prev_training_error = te_dfa[-1] if te_dfa else 0
        if np.abs(training_error - prev_training_error) <= tol:
            te_dfa.append(training_error)
            print(f'Hitting tolerance of {tol} with {np.abs(training_error - prev_training_error)}')
            break
        te_dfa.append(training_error)
        loss_dfa.append(epoch_loss)

    return te_dfa, loss_dfa, angles


# ====================================================================================================================================




# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Dataset stuff 
    # # ======================== MNIST ============================
    # from tensorflow.keras.datasets import mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train.astype(np.float32)
    # X_test = X_test.astype(np.float32)
    # X_train /= 255.0
    # X_test /= 255.0
    # X_train = X_train.reshape(60000, 28 * 28)
    # X_test = X_test.reshape(10000, 28 * 28)
    # nb_classes = 10
    # y_train = np.eye(nb_classes)[y_train]
    # y_test = np.eye(nb_classes)[y_test]
    # # /!\ DO NOT FORGET TO *UN*COMMENT IMSHOW IN TEST /!\

    # # Define training parameters
    # tol = 1e-4
    # batch_size = 200
    # learning_rate = 1e-4
    # epochs = 2

    # # Define the model
    # input_size = 784 #784
    # hidden_size = 80 #800
    # output_size = 10 #10

    # # Activation function and its derivative
    # act_function = nn.Tanh()
    # def act_function_derivative(x):
    #     return 1 - torch.tanh(x) ** 2

    # # Plot options
    # labels = [str(i) for i in range(10)]
    
    # model = DynamicModel([input_size, hidden_size, hidden_size, hidden_size, output_size], act_function, act_function_derivative)

    # # ======================== MNIST ============================




    # # ===============================================================================
    X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # XOR inputs
    y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR targets
    X_test = X_train
    y_test = y_train
    # /!\ DO NOT FORGET TO COMMENT IMSHOW IN TEST /!\

    # Define training parameters
    tol = -1e-4
    batch_size = 1
    learning_rate = 1e-1
    epochs = 1000

    # Define the model
    input_size = 2 #784
    hidden_size = 4 #800
    output_size = 1 #10

    # Activation function and its derivative
    act_function = nn.Tanh()
    def act_function_derivative(x):
        return 1 - torch.tanh(x) ** 2

    # # Plot options
    labels = ["v"]

    model = DynamicModel([input_size, hidden_size, hidden_size, hidden_size, output_size], act_function, act_function_derivative)
    # # ===============================================================================



    # Train the model
    te_dfa, loss_dfa, angles = train(model, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol = tol)


    # 
    plt.plot(range(len(te_dfa)), te_dfa, label='DFA')
    plt.title(f'Training error (lr:{learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Training error %')
    plt.legend(loc='best')
    plt.show()

    plt.plot(range(len(loss_dfa)), loss_dfa, label='DFA')
    plt.title(f'Loss (lr:{learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

    # If you need class indices for your model (assuming model output is a class index)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_indices = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # Testing loop
    from utils import plot_probas
    inp = "0"
    while inp != "q":
        X = torch.tensor(X_test[int(inp)], dtype=torch.float32)
        print(y_test[int(inp)])
        pred = model.forward_pass_light(X)
        print(f'Model result  : {pred}')
        plot_probas(pred.detach().numpy(), labels)
        predicted_val = np.argmax(pred.detach().numpy())
        print(f'Expected pred : {predicted_val}')
        # plt.imshow(X.reshape(28, 28))
        # plt.show()
        inp = input("\n> ")
