import matplotlib.pyplot as plt
from .dfa_step import dfa_update_step


def train(model, data, targets, loss_fn, learning_rate=0.01, epochs=1000):
    # Initialize feedback matrices
    Bs = model.Init_feedback()

    losses = []  # To store the loss at each epoch
    for epoch in range(epochs):
        # Forward pass

        # Pass data through each hidden layer and store activations
        hidden_activations = []
        x = data
        for layer in model.layers[:-1]:  # Collect activations for hidden layers only
            x = model.activation(layer(x))
            hidden_activations.append(x)

        output, loss = model.forward_pass(data, targets, loss_fn)

        # Apply DFA update step
        dfa_update_step(model, Bs, output, targets, hidden_activations, data, learning_rate)

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
