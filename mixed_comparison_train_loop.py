from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy



"""


Loop used to compare angles of gradients between a DFA and a Backprop step. Used in `main_angles.py`


"""


def compute_angles(model1, model2):
    angles = []

    # Ensure both models have the same structure
    for (param1, param2) in zip(model1.parameters(), model2.parameters()):
        # Flatten the weights into vectors
        vec1 = param1.view(-1)
        vec2 = param2.view(-1)

        # Compute the dot product and magnitudes
        dot_product = torch.dot(vec1, vec2)
        magnitude1 = torch.norm(vec1)
        magnitude2 = torch.norm(vec2)

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            angle = float('nan')  # Undefined if one vector is zero
        else:
            # Compute the cosine of the angle and take arccos
            cos_theta = dot_product / (magnitude1 * magnitude2)
            # Clamp to avoid numerical issues (cosine values slightly outside [-1, 1])
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle = torch.acos(cos_theta).item()

        angles.append(angle)

    return np.array(angles)









def train_mix(model, x, y, n_epochs=10, lr=1e-3, batch_size=200, tol=1e-1):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize random feedback matrix
    B = [torch.randn(layer.out_features, model.layers[-1].out_features) for layer in model.layers[:-1]]

    # Dataset conversion stuff
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    back_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    back_criterion = torch.nn.MSELoss()  # Replace with appropriate loss for your task



    # Recorded metrics
    te_dfa = []
    loss_dfa = []
    angles = []

    te_back = []
    loss_back = []
    # Training loop
    for epoch in range(n_epochs):
        dfa_epoch_loss = 0.0
        dfa_train_error = 0

        back_epoch_loss = 0.0
        back_train_error = 0.0

        angle_list = np.array([0.0 for _ in range(len(model.layers) * 2)]) # end up doing weight then bias angles

        # Batch run
        for batch_x, batch_y in dataloader:


            # ================  D F A  ============================
            # Forward pass
            dfa_a, dfa_h, dfa_y_hat = model.forward_pass_train(batch_x)

            # Error and error metric
            dfa_error = dfa_y_hat - batch_y
            dfa_preds = torch.argmax(dfa_y_hat, dim=1)
            dfa_truth = torch.argmax(batch_y, dim=1)
            dfa_train_error += (dfa_preds != dfa_truth).sum().item()

            # Loss metric
            # loss_on_batch = nn.CrossEntropyLoss()(y_hat, batch_y)
            dfa_loss_on_batch = F.binary_cross_entropy(dfa_y_hat, batch_y)
            dfa_epoch_loss += dfa_loss_on_batch.item()

            # Transposition of a and h for dimensions match
            dfa_a = [matrix.T for matrix in dfa_a]
            dfa_h = [matrix.T for matrix in dfa_h]

            # DFA backward pass
            dfa_dW, dfa_db= model.dfa_backward_pass(dfa_error.T, dfa_h, dfa_a, B)
            


            # Update weights manually
            with torch.no_grad():
                for i in range(len(dfa_db)):
                    model.layers[i].weight += lr * dfa_dW[i]
                    model.layers[i].bias += lr * dfa_db[i].squeeze()

            

            # ================  B A C K P R O P  ============================
            model_back = copy.deepcopy(model)
            back_outputs = model_back(batch_x)
            back_loss = back_criterion(back_outputs, batch_y)

            back_preds = torch.argmax(back_outputs, dim=1)
            back_truth = torch.argmax(batch_y, dim=1)
            back_train_error += (back_preds != back_truth).sum().item()

            # Backward pass
            back_optimizer.zero_grad()
            back_loss.backward()
            back_optimizer.step()

            # Accumulate batch loss
            back_epoch_loss += back_loss.item() * batch_x.size(0)

            # Accumulate angle
            angle_list += compute_angles(model, model_back)

        angles.append(angle_list)

        dfa_training_error = dfa_train_error / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {dfa_epoch_loss / len(dataloader):.4f}, Training Error = {dfa_training_error:.4f}, debug angle = {angles[-1][0]}")



        te_dfa.append(dfa_training_error)
        loss_dfa.append(dfa_epoch_loss)


        back_epoch_loss /= len(x)
        back_train_error /= len(x)
        loss_back.append(back_epoch_loss)
        te_back.append(back_train_error)

    return {
        "te_dfa": te_dfa,
        "te_back": te_back,
        "loss_dfa": loss_dfa,
        "loss_back": loss_back,
        "angles": angles
    }
