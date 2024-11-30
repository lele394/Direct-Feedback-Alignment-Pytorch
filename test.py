import numpy as np

import numpy as np

def plot_probas(X, labels, graph_height=5, gap=1, fill_char="█", space_char=" "):
    """
    Make sure X has the same size as labels, and values are between 0 and 1.
    """

    if len(X) != len(labels): print(f'PLOT_ERROR : Labels is of size ({len(labels)}) but expected of same size as X ({len(X)})'); return
    if max(X) > 1.0: print(f'PLOT_ERROR : X has a max value of {max(X)}, please use max 1'); return
    if min(X) < 0.0: print(f'PLOT_ERROR : X has a max value of {min(X)}, please use min 0'); return

    fill_perc = [np.floor(val * graph_height) / graph_height for val in X]

    graph = np.zeros((graph_height, len(X)))
    for i in range(len(labels)):
        for j in range(int(fill_perc[i]*graph_height)):
            graph[-j-1, i] = 1
    graph_str = ""
    for row in graph:
        graph_str += "".join(fill_char+space_char*gap if cell == 1 else space_char+space_char*gap for cell in row) + "\n"

    max_label = max([len(i) for i in labels])
    for slice in range(max_label):
        graph_str += "".join((label[slice] if slice < len(label) else space_char) + space_char * gap for label in labels) + "\n"

    print(graph_str)
    return



test_val = [np.random.random() for _ in range(6)]
# print(test_val)
labels = ["dog", "cat", "horse", "muppet", "dolphin", "monkey"]
# print(labels)


plot_probas(test_val, labels, graph_height=8, gap=4)