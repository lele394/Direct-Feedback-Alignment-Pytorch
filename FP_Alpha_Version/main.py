import torch
import copy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import numpy as np
from scipy.special import expit


class DynamicModel(nn.Module):
    def __init__(self, layer_sizes, act_function):
        super(DynamicModel, self).__init__()
        # Define layers using ModuleList
        self.layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]
        )
        self.act_function = act_function  # Activation function

    def forward(self, x):
        # Forward pass through each layer
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all except the last layer
            if i < len(self.layers) - 1:
                x = self.act_function(x)
        return x









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

# Forward pass using PyTorch model
def forward_pass(model, x):
    h1 = model.act_function(model.layers[0](x))
    a1 = model.layers[0](x)
    a2 = model.layers[1](h1)
    y_hat = torch.sigmoid(a2)
    return a1, h1, a2, y_hat

# DFA backward pass with PyTorch tensors
def dfa_backward_pass(e, h1, B1, a1, x):
    dW2 = -torch.matmul(e, h1.T)
    da1 = torch.matmul(B1, e) * (1 - torch.tanh(a1) ** 2)
    dW1 = -torch.matmul(da1, x.T)
    db1 = -torch.sum(da1, dim=1, keepdim=True)
    db2 = -torch.sum(e, dim=1, keepdim=True)
    return dW1, dW2, db1, db2

# Training loop
def train_DFA(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize random feedback matrix
    B1 = torch.randn(model.layers[0].out_features, model.layers[-1].out_features)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    angles = []
    te_dfa = []
    loss_dfa = []
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        train_error = 0

        for batch_x, batch_y in dataloader:
            # Forward pass
            a1, h1, a2, y_hat = forward_pass(model, batch_x)
            error = y_hat - batch_y
            preds = torch.argmax(y_hat, dim=1)
            truth = torch.argmax(batch_y, dim=1)
            train_error += (preds != truth).sum().item()

            # Calculate loss
            loss_on_batch = F.binary_cross_entropy(y_hat, batch_y)
            epoch_loss += loss_on_batch.item()

            # DFA backward pass
            dW1, dW2, db1, db2 = dfa_backward_pass(error.T, h1.T, B1, a1.T, batch_x.T)

            # print(dW1.shape)


            # Update weights manually
            with torch.no_grad():
                model.layers[0].weight += lr * dW1
                model.layers[0].bias += lr * db1.squeeze()
                model.layers[1].weight += lr * dW2
                model.layers[1].bias += lr * db2.squeeze()

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






def train_BACK(model, x, y, n_epochs=10, lr=1e-3, batch_size=32, tol=1e-4):
    """
    Train a PyTorch model using backpropagation.

    Parameters:
        model (torch.nn.Module): The model to train.
        X_train (torch.Tensor): Input training data.
        y_train (torch.Tensor): Target training labels.
        n_epochs (int): Number of epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        tol (float): Tolerance for early stopping.

    Returns:
        tuple: Training error (list of epoch losses) and final loss.
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader for batch processing
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # Replace with appropriate loss for your task

    # Training loop
    train_errors = []
    epoch_losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        train_error = 0.0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            preds = torch.argmax(outputs, dim=1)
            truth = torch.argmax(batch_y, dim=1)
            train_error += (preds != truth).sum().item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item() * batch_X.size(0)

        # Average epoch loss
        epoch_loss /= len(X_train)
        train_error /= len(X_train)
        epoch_losses.append(epoch_loss)
        train_errors.append(train_error)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping criterion
        if len(train_errors) > 1 and abs(train_errors[-2] - train_errors[-1]) < tol:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_errors, epoch_losses






# Example usage
if __name__ == "__main__":

    # Dataset stuff 
    # # ======================== MNIST ============================
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0
    X_train = X_train.reshape(60000, 28 * 28)
    X_test = X_test.reshape(10000, 28 * 28)
    nb_classes = 10
    y_train = np.eye(nb_classes)[y_train]
    y_test = np.eye(nb_classes)[y_test]
    # # ======================== MNIST ============================

    # ===============================================================================
    # X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # XOR inputs
    # y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR targets
    # ===============================================================================


    # Define the model
    input_size = 784 #784
    hidden_size = 200 #800
    output_size = 10 #10
    epochs = 100
    batch_size = 200
    learning_rate = 1e-4
    act_function = nn.Tanh()

    model_DFA = DynamicModel([input_size, hidden_size, output_size], act_function)
    model_BACK = copy.deepcopy(model_DFA)
    # Dummy data
    # x_train = np.random.rand(1000, input_size).astype(np.float32)
    # y_train = np.eye(output_size)[np.random.randint(0, output_size, 1000)].astype(np.float32)

    # Train the model using DFA
    te_dfa, loss_dfa, angles = train_DFA(model_DFA, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol = 1e-4)

    # Train using Backprop
    # error with the loss on this one /shrug
    te_back, loss_back = train_BACK(model_BACK, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol=1e-4)





    import matplotlib.pyplot as plt
    plt.plot(range(len(te_dfa)), te_dfa, label='DFA training error')
    plt.plot(range(len(te_back)), te_back, label='Backprop training error')
    plt.title('Learning rate 1e-4')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.ylabel('Training error %')
    plt.legend(loc='best')
    plt.show()



    # import matplotlib.pyplot as plt
    # plt.plot(range(len(loss_dfa)), loss_dfa, label='DFA Loss')
    # plt.plot(range(len(loss_back)), loss_back, label='Backprop loss')
    # plt.title('Learning rate 1e-4')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend(loc='best')
    # plt.show()






    # If you need class indices for your model (assuming model output is a class index)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_indices = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # Test the trained model



    # print(f"Test Error Rate: {test_error:.2%}")




    from utils import plot_probas
    labels = [str(i) for i in range(10)]
    inp = "0"
    while inp != "q":
        X = torch.tensor(X_test[int(inp)], dtype=torch.float32)
        print(y_test[int(inp)])
        a1, h1, a2, pred = forward_pass(model_DFA, X)
        print(f'Model result  : {pred}')
        plot_probas(pred.detach().numpy(), labels)
        predicted_val = np.argmax(pred.detach().numpy())
        print(f'Expected pred : {predicted_val}')
        plt.imshow(X.reshape(28, 28))
        plt.show()
        inp = input("\n> ")
