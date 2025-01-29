from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




def MNIST_error(y_hat, batch_y, train_error):
    """
    MNIST_error function calculates the error between predicted and true labels for a batch of data.
    It computes the difference between predicted and true values, determines if the predictions are correct,
    and updates the total training error.

    Parameters:
        y_hat (torch.Tensor): The predicted values from the model (logits).
        batch_y (torch.Tensor): The true labels (one-hot encoded).
        train_error (float): The cumulative training error.

    Returns:
        error (torch.Tensor): The error between predictions and true values.
        train_error (float): The updated cumulative training error.
    """
    error = y_hat - batch_y
    preds = torch.argmax(y_hat, dim=1)
    truth = torch.argmax(batch_y, dim=1)
    train_error += (preds != truth).sum().item()

    return error, train_error


def train_basic(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1, error_function=MNIST_error):
    """
    Train a basic neural network model using Direct Feedback Alignment.
    This function handles the training loop, including the forward pass, error computation, loss calculation, 
    and weight updates. It also implements a tolerance check for early stopping based on the training error.

    Arguments:
        model (torch.nn.Module): The neural network model to be trained.
        x (array-like): The input features for the training dataset.
        y (array-like): The target labels for the training dataset.
        n_epochs (int, optional): The number of epochs to train the model (default is 10).
        lr (float, optional): The learning rate for weight updates (default is 1e-3).
        batch_size (int, optional): The batch size used for training (default is 200).
        tol (float, optional): The tolerance value for early stopping based on training error (default is 1e-1).
        error_function (function, optional): A function to calculate the error metric (default is MNIST_error).

    Returns:
        te_dfa (list): The list of training errors recorded at each epoch.
        loss_dfa (list): The list of loss values recorded at each epoch.
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)


    def matrix_transfo(x):
        return x

    B = [matrix_transfo(torch.empty(layer.out_features, model.layers[-1].out_features).normal_(mean=0,std=1.9)) for layer in model.layers[:-1]]

    # Dataset conversion stuff
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Recorded metrics
    te_dfa = []
    loss_dfa = []

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        train_error = 0

        # Batch run
        for batch_x, batch_y in dataloader:
            # Forward pass
            a, h, y_hat = model.forward_pass_train(batch_x)

            # Error and error metric
            error, train_error = error_function(y_hat, batch_y, train_error)

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


        training_error = train_error / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}, Training Error = {training_error:.4f}")

        prev_training_error = te_dfa[-1] if te_dfa else 0
        if np.abs(training_error - prev_training_error) <= tol:
            te_dfa.append(training_error)
            print(f'Hitting tolerance of {tol} with {np.abs(training_error - prev_training_error)}')
            break
        te_dfa.append(training_error)
        loss_dfa.append(epoch_loss)

    return te_dfa, loss_dfa