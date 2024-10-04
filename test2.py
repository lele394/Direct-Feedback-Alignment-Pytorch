import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

# Dataset generation
def CreateSpiral(n_points=1000, start_theta=0, expansion_rate=0.2, turns=2):
    theta = np.linspace(0, turns * 2 * np.pi, n_points)
    r = expansion_rate * theta
    x = r * np.cos(theta + start_theta)
    y = r * np.sin(theta + start_theta)
    return x, y

n_points = 5000
x1, y1 = CreateSpiral(n_points)
x2, y2 = CreateSpiral(n_points, start_theta=np.pi)

# Plot spirals
if False:
    plt.scatter(x1, y1, label="spiral 1")
    plt.scatter(x2, y2, label="spiral 2")
    plt.legend()
    plt.show()

# Full dataset
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
dataset = np.array([[x[i], y[i]] for i in range(len(x))])

expected_results = np.array([[1, 0] for i in range(n_points)] + [[0, 1] for i in range(n_points)])

# Define the model
class DFAModel(tf.keras.Model):
    def __init__(self):
        super(DFAModel, self).__init__()
        self.dense1 = layers.Dense(50, activation='relu')
        self.dense2 = layers.Dense(50, activation='relu')
        self.dense3 = layers.Dense(50, activation='relu')
        self.output_layer = layers.Dense(2)
        
        # Random feedback weights for DFA
        self.feedback1 = tf.random.normal((2, 50))
        self.feedback2 = tf.random.normal((2, 50))
        self.feedback3 = tf.random.normal((2, 50))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)







# Custom DFA weight update function
def custom_backprop(model, inputs, targets, learning_rate=0.01):

    # CURRENTLY NOT DFA BUT BACKPROP FOR TESTING PURPOSES

    # Define the loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        # Forward pass through the model
        outputs = model(inputs)  # Forward pass through the entire model
        loss = loss_fn(targets, outputs)  # Compute the loss

    # Compute gradients for all trainable variables in the model
    gradients = tape.gradient(loss, model.trainable_variables)

    # Update weights for all layers using backpropagation
    for var, grad in zip(model.trainable_variables, gradients):
        var.assign_sub(learning_rate * grad)











def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)





# Custom DFA weight update function
def custom_dfa_training_step(model, inputs, targets, learning_rate=0.01):


    # ========================== FORWARD PASS =======================================

    # layer 1
    a1 = np.dot(inputs, model.dense1.kernel) + model.dense1.bias
    z1 = np.maximum(0, a1)  # ReLU activation 

    # layer 2
    a2 = np.dot(z1, model.dense2.kernel) + model.dense2.bias
    z2 = np.maximum(0, a2)  # ReLU activation 

    # layer 3 
    a3 = np.dot(z2, model.dense3.kernel) + model.dense3.bias
    z3 = np.maximum(0, a3)  # ReLU activation 

    # output layer
    aO = np.dot(z3, model.output_layer.kernel) + model.output_layer.bias
    zO = np.maximum(0, aO)  # ReLU activation 


    # ======================== RANDOM WEIGHTS MATRICES B_i =========================

    B_O = np.random.normal(size=model.output_layer.kernel.shape)
    B_3 = np.random.normal(size=model.dense3.kernel.shape)
    B_2 = np.random.normal(size=model.dense2.kernel.shape)
    B_1 = np.random.normal(size=model.dense1.kernel.shape)

    # ======================== ERROR e =============================================

    e = zO - targets
    print(e)



























# Instantiate model
dfa_model = DFAModel()

# Forward pass with dummy data to build the model
dummy_input = tf.random.normal((1, dataset.shape[1]))  # Match the shape of your input data
_ = dfa_model(dummy_input)  # Perform a forward pass to build the model


# Training loop
epochs = 400
batch_size = 200
learning_rate = 0.01

dataset = tf.convert_to_tensor(dataset, dtype=tf.float32)
expected_results = tf.convert_to_tensor(expected_results, dtype=tf.float32)

for epoch in range(epochs):
    print(f'epcoh: {epoch}')
    for i in range(0, len(dataset), batch_size):
        x_batch = dataset[i:i + batch_size]
        y_batch = expected_results[i:i + batch_size]
        custom_dfa_training_step(dfa_model, x_batch, y_batch, learning_rate)






# Evaluate the model
predictions = dfa_model(dataset)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(expected_results, axis=1))
print(f"Accuracy: {accuracy}")

# Test on noisy testset
noise = 0.22
xt1, yt1 = CreateSpiral(200)
xt2, yt2 = CreateSpiral(200, start_theta=np.pi)
xt = np.concatenate((xt1, xt2))
yt = np.concatenate((yt1, yt2))

testset = np.array([[xt[i] + np.random.rand() * noise, yt[i] + np.random.rand() * noise] for i in range(len(xt))])


# Prediction square 
pred_square = []
for a in range(100):
    for b in range(100):
        pred_square.append([(a/100)*4-2,(b/100)*4-2])
testset = np.array(pred_square) # comment to disable square test



test_predictions = dfa_model(testset)

# Classify and plot the results
spiral1 = []
spiral2 = []
spiralUnk = []

for i in range(len(test_predictions)):
    pred = test_predictions[i]
    if pred[0] > 0.5 and pred[1] < 0.5:
        spiral1.append(testset[i])
    elif pred[0] < 0.5 and pred[1] > 0.5:
        spiral2.append(testset[i])
    else:
        spiralUnk.append(testset[i])

def PlotSpiral(data, label, style="bo"):
    x = [dat[0] for dat in data]
    y = [dat[1] for dat in data]
    plt.plot(x, y, style, label=label)

PlotSpiral(spiral1, "Spiral 1", "b.")
PlotSpiral(spiral2, "Spiral 2", "r.")
PlotSpiral(spiralUnk, "Unknown", "g.")

plt.legend()
plt.show()

dfa_model.summary()

