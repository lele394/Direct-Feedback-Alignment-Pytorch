import torch

def dfa_update_step(model, Bs, output, target, hidden_activations, inputs, learning_rate=0.01):
    # Compute error at the output
    error = output - target  # Compute error at the output layer

    # Update the output layer first
    feedback_signal_output = error * model.activation_derivative(output)
    with torch.no_grad():
        # Update output layer weights and biases
        delta_w_out = torch.matmul(feedback_signal_output.T, hidden_activations[-1])
        model.layers[-1].weight -= learning_rate * delta_w_out
        delta_b_out = feedback_signal_output.sum(0)
        model.layers[-1].bias -= learning_rate * delta_b_out

    # Now, propagate backward through the hidden layers
    for index in reversed(range(len(model.layers) - 1)):  # Hidden layers (in reverse order)
        # Compute feedback signal for hidden layers
        feedback_signal_hidden = torch.matmul(feedback_signal_output, Bs[index]) * model.activation_derivative(hidden_activations[index])

        with torch.no_grad():
            # Update weights and biases for hidden layers
            delta_w = torch.matmul(feedback_signal_hidden.T, hidden_activations[index - 1] if index > 0 else inputs)
            model.layers[index].weight -= learning_rate * delta_w
            delta_b = feedback_signal_hidden.sum(0)
            model.layers[index].bias -= learning_rate * delta_b

            # Prepare feedback signal for the next (previous) layer
            feedback_signal_output = feedback_signal_hidden

