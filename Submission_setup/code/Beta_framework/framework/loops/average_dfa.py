from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def train_averaged(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    """
    Function to train a model using an averaged feedback mechanism. The training process involves 
    forward and backward passes, including the computation of error metrics and the loss function 
    using binary cross-entropy. The feedback matrix is updated after each batch and averaged 
    throughout the training loop. The model's weights and biases are manually updated using the 
    computed gradients. The training continues for a specified number of epochs or until the 
    training error converges based on a tolerance value.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        x (numpy.ndarray or torch.Tensor): Input features for training.
        y (numpy.ndarray or torch.Tensor): Target labels for training.
        n_epochs (int, optional): The number of epochs to train for (default is 10).
        lr (float, optional): The learning rate for weight updates (default is 1e-3).
        batch_size (int, optional): The number of samples per batch (default is 200).
        tol (float, optional): The tolerance value for early stopping based on training error (default is 1e-1).

    Returns:
        te_dfa (list): A list of training errors recorded at each epoch.
        loss_dfa (list): A list of the loss values recorded at each epoch.
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize random feedback matrix
    B = [torch.randn(layer.out_features, model.layers[-1].out_features) for layer in model.layers[:-1]]

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


            # print('a : ', a,'h : ', h,'y : ', y_hat)
            # print(type(a), type(h), type(y_hat))
            # quit()


            # Error and error metric
            error = y_hat - batch_y # used only for metric, recomputed later using averaged
            preds = torch.argmax(y_hat, dim=0)
            truth = torch.argmax(batch_y, dim=0)
            train_error += (preds != truth).sum().item()

            # Loss metric
            # loss_on_batch = nn.CrossEntropyLoss()(y_hat, batch_y)
            loss_on_batch = F.binary_cross_entropy(y_hat, batch_y)
            epoch_loss += loss_on_batch.item()

            # print(a)
            # print(h)
            # quit()

            # ======== averaging stuff ============
            for i in range(len(a)):
                a[i] = torch.mean(a[i], dim=0).unsqueeze(0)

            for i in range(len(h)):
                h[i] = torch.mean(h[i], dim=0).unsqueeze(0)

            y_hat = torch.mean(y_hat, dim=0).unsqueeze(0)
            batch_y = torch.mean(batch_y, dim=0).unsqueeze(0)
            batch_x = torch.mean(batch_x, dim=0).unsqueeze(0)
            # ====================================

            # Recompute error using averaged batches
            error = y_hat - batch_y

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

