import torch
import torch.nn as nn
import copy

import numpy as np
from scipy.special import expit


#============

from framework import DynamicModel

#============

"""
Notes to future me or other users.

The following does not support the use of custom activation functions for each layer (low-key want to add that though. I might).

"""

# ====================================================================================================================================
# Training loop
# from framework import train_basic as train
# from framework import MNIST_train_class_averaged as train
from framework import Backprop_train as train_BACK
# ====================================================================================================================================





def average_angle(W2, B1, error, a1, a2):
    dh1 = np.mean(np.matmul(B1, error), axis=1, keepdims=True)  # Maybe no derivative needed here
    c1 = np.mean(np.matmul(np.transpose(W2), error * (expit(a2) * (1 - expit(a2)))), axis=1, keepdims=True)
    dh1_norm = np.linalg.norm(dh1)
    c1_norm = np.linalg.norm(c1)
    inverse_dh1_norm = np.power(dh1_norm, -1)
    inverse_c1_norm = np.power(c1_norm, -1)
    
    # ALIGNMENT CRITERION AND ANGLE
    Lk = (np.matmul(np.transpose(dh1), c1) * inverse_dh1_norm)[0, 0]
    beta = np.arccos(np.clip(Lk * inverse_c1_norm, -1., 1.)) * 180 / np.pi
    return Lk, beta








# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    Test_DS = "MNIST"


    # Dataset stuff 
    # # ======================== MNIST ============================

    from tensorflow.keras.datasets import mnist
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
    tol = -1
    batch_size = 500
    learning_rate = 1e-5
    epochs = 2000

    # Define the model
    input_size = 784 #784
    hidden_size = 80 #800
    output_size = 10 #10

    # Activation function and its derivative
    act_function = nn.Tanh()
    def act_function_derivative(x):
        return 1 - torch.tanh(x) ** 2

    # Plot options
    labels = [str(i) for i in range(10)]

    def output(inp):
        foo=nn.Softmax(dim=0)(inp)
        return foo

    # # ======================== MNIST ============================
    model_DFA = DynamicModel([input_size, hidden_size, hidden_size, output_size], act_function, act_function_derivative)
    model_BACK = copy.deepcopy(model_DFA)

    # Train the model
    print("\n\n ============== DFA TRAINING ===================")
    te_dfa, loss_dfa = train(model_DFA, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol = tol)
    
    print("\n\n ============ BACKPROP TRAINING ===================")
    te_back, loss_back = train_BACK(model_BACK, X_train, y_train, n_epochs=epochs, lr=learning_rate, batch_size=batch_size, tol=tol)



    # 
    plt.plot(range(len(te_dfa)), te_dfa, label='DFA')
    plt.plot(range(len(te_back)), te_back, label='Backprop')
    plt.title(f'Training error (lr:{learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Training error %')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()

    plt.plot(range(len(loss_dfa)), loss_dfa, label='DFA')
    plt.plot(range(len(loss_back)), loss_back, label='Backprop')
    plt.title(f'Loss (lr:{learning_rate})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.show()




    nb_dat = len(X_test)
    found = 0
    for index in range(nb_dat):
        X = torch.tensor(X_test[index], dtype=torch.float32)
        pred = model_DFA(X)

        predicted_val = np.argmax(pred.detach().numpy())
        expected_val = np.argmax(y_test[index])
        if expected_val == predicted_val:
            found += 1
    print(f'\n\nNb dat : {nb_dat}\t found : {found}\t Accuracy : {found/nb_dat}')


    nb_dat = len(X_test)
    found = 0
    for index in range(nb_dat):
        X = torch.tensor(X_test[index], dtype=torch.float32)
        pred = model_BACK(X)

        predicted_val = np.argmax(pred.detach().numpy())
        expected_val = np.argmax(y_test[index])
        if expected_val == predicted_val:
            found += 1
    print(f'\n\nNb dat : {nb_dat}\t found : {found}\t Accuracy : {found/nb_dat}')
    # ============================================================
    # Currently discarded
    # ============================================================
    # # If you need class indices for your model (assuming model output is a class index)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test_indices = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # # Testing loop
    # from utils import plot_probas
    # inp = "0"
    # while inp != "q":
    #     X = torch.tensor(X_test[int(inp)], dtype=torch.float32)
    #     print(y_test[int(inp)])
    #     pred = model.forward_pass_light(X)
    #     print(f'Model result  : {pred}')
    #     plot_probas(pred.detach().numpy(), labels)
    #     predicted_val = np.argmax(pred.detach().numpy())
    #     print(f'Expected pred : {predicted_val}')
    #     # plt.imshow(X.reshape(28, 28))
    #     # plt.show()
    #     inp = input("\n> ")
