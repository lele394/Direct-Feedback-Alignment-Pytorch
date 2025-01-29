from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def MNIST_train_class_averaged_LR_scheduler(model, x, y, lr_schedule, n_epochs=10, batch_size=200, tol=1e-1, classes=10):
    """
    This function implements a training loop for a model on the MNIST dataset using a custom class-averaged
    learning rate scheduler. The key features include:

    - `lr_schedule`: A function that defines the learning rate for each epoch.
    - Forward and backward passes with class-specific averaging of activations and outputs.
    - Direct Feedback Alignment (DFA) for weight updates, utilizing a random feedback matrix `B`.
    - Tracks training loss and error metrics across epochs.
    - Early stopping based on a user-defined tolerance (`tol`) for error changes.
    Here, `lr_schedule` is a function that returns the learning rate in function of the epoch
    eg : lr=lr_schedule(epoch)
    
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

        lr=lr_schedule(epoch)

        

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
        print(f"Epoch {epoch+1}/lr:{lr:.2f}: Loss = {epoch_loss / len(dataloader):.4f}, Training Error = {training_error:.4f}")

        prev_training_error = te_dfa[-1] if te_dfa else 0
        if np.abs(training_error - prev_training_error) <= tol:
            te_dfa.append(training_error)
            print(f'Hitting tolerance of {tol} with {np.abs(training_error - prev_training_error)}')
            break
        te_dfa.append(training_error)
        loss_dfa.append(epoch_loss)

    return te_dfa, loss_dfa