import torch
import torch.nn as nn
import copy

import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt

#============

from model import DynamicModel

#============

"""
Notes to future me or other users.

The following does not support the use of custom activation functions for each layer (low-key want to add that though. I might).

"""

# ====================================================================================================================================
# Training loop
from dfa_train_loop import train_basic as train
# ====================================================================================================================================

# Dataset stuff 
# # ======================== XOR ============================

X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # XOR inputs
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)  # XOR targets
X_test = X_train
y_test = y_train
# /!\ DO NOT FORGET TO *UN*COMMENT IMSHOW IN TEST /!\


# Define training parameters
tol = -1
batch_size = 1
learning_rate = 1e-1
epochs = 1000

# Define the model
input_size = 2 #784
hidden_size = 4 #800
output_size = 1 #10

# Activation function and its derivative
act_function = nn.Tanh()
def act_function_derivative(x):
    return 1 - torch.tanh(x) ** 2


def output(inp):
    foo=nn.Softmax(dim=0)(inp)
    return foo


def error_function(y_hat, batch_y, train_error):
    error = y_hat - batch_y
    train_error += np.abs(error.sum().item())
    return error, train_error





model_DFA = DynamicModel([input_size, hidden_size, output_size], act_function, act_function_derivative)
# # # ======================== MNIST ============================

# Train the model
te_dfa, loss_dfa = train(model_DFA, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol = -1, error_function=error_function)

# print(type(te_dfa))

plt.plot(range(len(te_dfa)), te_dfa, label='DFA')
plt.title(f'Training error (lr:{learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Training error')
plt.yscale('log')
plt.legend(loc='best')
plt.show()


plt.plot(range(len(loss_dfa)), loss_dfa, label='DFA')
plt.title(f'Loss (lr:{learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='best')
plt.show()


for x in X_test:
    print(f' {x}  :  {model_DFA(x)}')
