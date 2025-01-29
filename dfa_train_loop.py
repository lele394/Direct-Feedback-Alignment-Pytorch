from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




def MNIST_error(y_hat, batch_y, train_error):
    error = y_hat - batch_y
    preds = torch.argmax(y_hat, dim=1)
    truth = torch.argmax(batch_y, dim=1)
    train_error += (preds != truth).sum().item()

    return error, train_error


def train_basic(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1, error_function=MNIST_error):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)


    def matrix_transfo(x):
        return x
        
        # Hi, whoever is around here,
        # I'm intimately convinced that convergence rate
        # is linked to matrix initialization.
        # I'm not sure if the matrix shouuld evolve during
        # epochs, by skewing the distribution during training
        # But something is going on here I'm sure.

        # send it between 0,1
        # x_max = torch.max(x)
        # x = x/x_max

        # return x

        # test 1 : 
        # offset = 0.5
        # dilation = 2
        # order = 4
        # f = np.sin(((x+offset)*dilation)*np.pi/2)**order
        # scaling = 0.8
        # max = 1.0
        # offset = max-scaling
        # return f*scaling+offset

        # test 2:
        # return x**1.8


    # B = [matrix_transfo(torch.rand(layer.out_features, model.layers[-1].out_features)) for layer in model.layers[:-1]]

    B = [matrix_transfo(torch.empty(layer.out_features, model.layers[-1].out_features).normal_(mean=0,std=1.9)) for layer in model.layers[:-1]]
    # B = [matrix_transfo(torch.randn(layer.out_features, model.layers[-1].out_features)) for layer in model.layers[:-1]]

    # print(torch.max(B[0]))

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

            # print(error)

            # Loss metric
            # loss_on_batch = nn.CrossEntropyLoss()(y_hat, batch_y)
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










































def train_averaged(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
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
























# NOT FINISHED YET
# Modifying to fit MNIST dataset, aka 10 classes.

def MNIST_train_class_averaged(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1, classes=10):
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















































# NOT FINISHED YET
# Modifying to fit MNIST dataset, aka 10 classes.

def MNIST_train_class_averaged_LR_scheduler(model, x, y, lr_schedule, n_epochs=10, batch_size=200, tol=1e-1, classes=10):
    """
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
