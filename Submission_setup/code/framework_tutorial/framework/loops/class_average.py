from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def MNIST_train_class_averaged(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1, classes=10):
    """
    Description:
    Trains a given neural network model on the MNIST dataset using a class-averaged 
    Direct Feedback Alignment (DFA) approach. The function performs the following tasks:
    1. Initializes random feedback matrices for DFA.
    2. Converts the dataset into a PyTorch DataLoader for batched training.
    3. Computes loss and training error for each epoch.
    4. Implements class-based averaging of intermediate layer activations and predictions.
    5. Updates model weights using DFA with the averaged error signals.
    6. Stops training early if the training error tolerance is met.

    Parameters:
    - model (torch.nn.Module): Neural network model to train.
    - x (numpy.ndarray): Input data (features).
    - y (numpy.ndarray): Target labels (one-hot encoded).
    - n_epochs (int): Number of training epochs. Default is 10.
    - lr (float): Learning rate for weight updates. Default is 1e-3.
    - batch_size (int): Size of each training batch. Default is 200.
    - tol (float): Training error tolerance for early stopping. Default is 1e-1.
    - classes (int): Number of output classes. Default is 10.

    Returns:
    - te_dfa (list): List of training errors for each epoch.
    - loss_dfa (list): List of loss values for each epoch.

    Notes:
    - The function uses binary cross-entropy loss for optimization.
    - Class-based averaging is used to compute error signals, which are then 
    propagated backward using DFA.
    - Early stopping occurs if the change in training error between epochs 
    falls below the specified tolerance.

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

            # Error and error metric
            error = y_hat - batch_y # used only for metric, recomputed later using averaged
            preds = torch.argmax(y_hat, dim=0)
            truth = torch.argmax(batch_y, dim=0)
            train_error += (preds != truth).sum().item()

            # Loss metric
            # loss_on_batch = nn.CrossEntropyLoss()(y_hat, batch_y)
            loss_on_batch = F.binary_cross_entropy(y_hat, batch_y)
            epoch_loss += loss_on_batch.item()


            # print(batch_y)
            # print(y_hat)
            # print(a)

            # ======== averaging stuff ============

            # get batch indices relevant to the classes
            class_indices = [ [] for _ in range(classes)]
            for item in range(len(batch_y)):
                class_indices[np.argmax(batch_y[item])].append(item)

            # print(indices)
            # quit()

            # loop over all classes
            for indices in class_indices:

                if indices == []:
                    continue
                # print(indices)


                # ======== averaging over a subset ==========
                subset_a = []
                subset_h = []
                
                # Assuming indices is a list or tensor of indices to average over
                for i in range(len(a)):
                    subset_a.append(torch.mean(a[i][indices], dim=0).unsqueeze(0))

                for i in range(len(h)):
                    subset_h.append(torch.mean(h[i][indices], dim=0).unsqueeze(0))

                sub_y_hat = torch.mean(y_hat[indices], dim=0).unsqueeze(0)
                sub_batch_y = torch.mean(batch_y[indices], dim=0).unsqueeze(0)
                # batch_x = torch.mean(batch_x[indices], dim=0).unsqueeze(0)
                # ==========================================

                # Recompute error using averaged batches
                error = sub_y_hat - sub_batch_y

                # print('subset_a\n',subset_a)
                # Transposition of a and h for dimensions match
                sub_a = [matrix.T for matrix in subset_a]
                sub_h = [matrix.T for matrix in subset_h]


                # DFA backward pass
                dW, db = model.dfa_backward_pass(error.T, sub_h, sub_a, B)

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



