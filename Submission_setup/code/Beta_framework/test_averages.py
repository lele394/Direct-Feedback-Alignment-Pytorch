import torch
import torch.nn as nn
from torchsummary import summary

import copy

import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt

# enable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#============

from framework import DynamicModel



#============

"""
Notes to future me or other users.

The following does not support the use of custom activation functions for each layer (low-key want to add that though. I might).

"""

# ====================================================================================================================================
# Training loop
# from dfa_train_loop import train_averaged as train 
from framework import MNIST_train_class_averaged as train 
# from dfa_train_loop import MNIST_train_class_averaged_LR_scheduler as train 

# ====================================================================================================================================



# Test_DS = "MNIST"


# Dataset stuff 
# # ======================== MNIST ============================

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train /= 255.0
X_test /= 255.0
X_train = X_train.reshape(60000, 28 * 28)
X_test = X_test.reshape(10000, 28 * 28)
nb_classes = 10
y_train = np.eye(nb_classes)[y_train]
y_test = np.eye(nb_classes)[y_test]
# /!\ DO NOT FORGET TO *UN*COMMENT IMSHOW IN TEST /!\








# Define training parameters
tol = 1e-4
batch_size = 1000
learning_rate = 1e-3
epochs = 80
# epochs = 2

# Define the model
input_size = 784 #784
hidden_size = 80 #800
output_size = 10 #10

# Activation function and its derivative
act_function = nn.Tanh()
def act_function_derivative(x):
    return 1 - torch.tanh(x) ** 2

# Sigmoid DOESNT WORK
# act_function = nn.Sigmoid()
# def sigmoid_derivative(x):
#     sigmoid = torch.Sigmoid(x)
#     return sigmoid * (1 - sigmoid)


# Plot options
labels = [str(i) for i in range(10)]

# def output(inp):
#     foo=nn.Softmax(dim=0)(inp)
#     return foo

model_DFA = DynamicModel([input_size, hidden_size, hidden_size, output_size], act_function, act_function_derivative) 
# Adding softmax break it, need to add derivate in backprop. , output_function=output)
# # # ======================== MNIST ============================

model_DFA.summary(input_shape=(1,input_size))



# def lr_schedule(epoch):
#     alpha = 10.8
#     beta = 4.6
#     return np.exp(-epoch/alpha - beta)

def lr_schedule(epoch):
    # Define the range and parameters for the sigmoid function
    min_lr = 0.001
    max_lr = 0.01
    dilation = 0.2
    offset = 35
    return min_lr + (max_lr - min_lr) * (1 / (1 + np.exp(-(epoch-offset)*dilation)))

if False:
    x = np.arange(0, epochs)
    y = [lr_schedule(ep) for ep in x]
    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Learning Rate Schedule')
    plt.title('Sigmoid Learning Rate Schedule between Epochs 0 and 80')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()




# Train the model
te_dfa, loss_dfa = train(model_DFA, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol = -1)
# te_dfa, loss_dfa = train(model_DFA, X_train, y_train, lr_schedule, n_epochs=epochs, batch_size=batch_size, tol = -1)

plt.plot(range(len(te_dfa)), te_dfa, label='DFA')
plt.title(f'Training error (lr:{learning_rate})')
plt.xlabel('Epochs')
plt.ylabel('Training error %')
plt.yscale('log')
plt.legend(loc='best')
plt.show()





# Test model
nb_dat = len(X_test)
found = 0
for index in range(nb_dat):
    X = torch.tensor(X_test[index], dtype=torch.float32)
    pred = model_DFA(X)

    predicted_val = np.argmax(pred.detach().numpy())
    expected_val = np.argmax(y_test[index])
    if expected_val == predicted_val:
        found += 1


# Save training error and loss
# np.array(te_dfa).tofile(f'LR_tests/TE_BATCH_CLASS_AVG_{batch_size}_{learning_rate}.dat')
# np.array(loss_dfa).tofile(f'LR_tests/LOSS_{learning_rate}.dat')



print(f'\n\nNb dat : {nb_dat}\t found : {found}\t Accuracy : {found/nb_dat}')
input(" << Continue >>")




# If you need class indices for your model (assuming model output is a class index)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_indices = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

# Testing loop
from utils import plot_probas
inp = "0"
while inp != "q":
    X = torch.tensor(X_test[int(inp)], dtype=torch.float32)
    print(y_test[int(inp)])
    pred = model_DFA(X)
    print(f'Model result  : {pred}')
    plot_probas(pred.detach().numpy(), labels)
    predicted_val = np.argmax(pred.detach().numpy())
    print(f'Model pred    : {predicted_val}')
    print(f'Expected pred : {np.argmax(y_test[int(inp)])}')

    # plt.imshow(X.reshape(28, 28))
    # plt.show()
    inp = input("\n> ")
