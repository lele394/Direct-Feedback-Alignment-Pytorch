import torch
from torch.utils.data import TensorDataset, DataLoader



def train(model, x, y, n_epochs=10, lr=1e-3, batch_size=32, tol=1e-4):
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
        epoch_loss /= len(x)
        train_error /= len(x)
        epoch_losses.append(epoch_loss)
        train_errors.append(train_error)

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.6f}, Error: {train_error:.6f}")

        # Early stopping criterion
        if len(train_errors) > 1 and abs(train_errors[-2] - train_errors[-1]) < tol:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_errors, epoch_losses

