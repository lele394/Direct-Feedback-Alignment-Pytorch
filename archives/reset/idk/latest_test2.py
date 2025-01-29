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
        # self.feedback1 = tf.random.normal((50, 2), dtype=tf.float32)  
        # self.feedback2 = tf.random.normal((50, 2), dtype=tf.float32)  
        # self.feedback3 = tf.random.normal((50, 2), dtype=tf.float32)  

        self.feedback_mat = [
             tf.random.normal((2,50), dtype=tf.float32) for _ in range(3) # (2,50) for dimension matching
        ]


    def call(self, inputs, no_feedback):
        # Forward pass
        a1 = self.dense1(inputs)  # Pre-activation output of layer 1
        h1 = tf.nn.relu(a1)  # Activation output of layer 1
        
        a2 = self.dense2(h1)  # Pre-activation output of layer 2
        h2 = tf.nn.relu(a2)  # Activation output of layer 2
        
        a3 = self.dense3(h2)  # Pre-activation output of layer 3
        h3 = tf.nn.relu(a3)  # Activation output of layer 3
        
        ANN_output = self.output_layer(h3)  # Final output



        if no_feedback:return ANN_output # Use this when dealing with backprop
        else:return ANN_output, a1, a2, a3, h1, h2, h3  # Return output and pre-activations







# Custom DFA weight update function
def custom_backprop(model, inputs, targets, learning_rate=0.01):

    # CURRENTLY NOT DFA BUT BACKPROP FOR TESTING PURPOSES


    # Define the loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    with tf.GradientTape() as tape:
        # Forward pass through the model
        outputs = model(inputs, no_feedback=True)  # Forward pass through the entire model
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

def relu(x):
    return max(0,x)









# ========================================== DFA HERE, SUPPOSEDLY ===============================================

# Custom DFA weight update function
def custom_dfa_training_step(model, inputs, targets, learning_rate=0.01):
    
    # Get batch size
    batch_size = inputs.shape[0]
    
    # 1. Forward pass to get the output from the model
    ANN_output, a1, a2, a3, h1, h2, h3 = model(inputs, no_feedback=False)
    
    # 2. Define the error
    error = ANN_output - targets  # Error as defined: ANN_output - targets


    # 3. Define random matrices for weight updates (feedback)
    # They are initialized in the model

    # 4. Update weights for each layer

    def ComputeLayerDeltaW(i_layer, a, error, h):
        B = model.feedback_mat[i_layer] # Random B matrix
        d_activation = relu_derivative(a)  # F' (here ReLu')
        delta_a = np.dot(error, B) * d_activation  # delta_a = (B.e)xF'(a)
        # Convert delta_a1 to float32 if necessary
        delta_a = tf.convert_to_tensor(delta_a, dtype=tf.float32)

        # Normalize by batch size
        delta_a /= batch_size  # Avoid exponential growth

        delta_a_transpose = tf.transpose(delta_a)
        delta_W = -tf.matmul(delta_a_transpose, h) # delta_w = -delta_a.h

        return delta_W 

    # 4.2
    delta_W1 = ComputeLayerDeltaW(0, a1, error, inputs)
    delta_W2 = ComputeLayerDeltaW(1, a2, error, h1)
    delta_W3 = ComputeLayerDeltaW(2, a3, error, h2)

    # 4.3 assigning weights
    weights1 = model.dense1.kernel 
    delta_W1 = learning_rate * delta_W1  # Clip the gradient updates
    weights1.assign(weights1 + tf.transpose(delta_W1))

    weights2 = model.dense2.kernel  
    delta_W2 = learning_rate * delta_W2  # Clip the gradient updates
    weights2.assign(weights2 + tf.transpose(delta_W2))

    weights3 = model.dense3.kernel  
    delta_W3 = learning_rate * delta_W3  # Clip the gradient updates
    weights3.assign(weights3 + tf.transpose(delta_W3))

    # Update for the output layer
    output_weights = model.output_layer.kernel  # Get the weights of the output layer
    delta_output_weights = -tf.matmul(tf.transpose(error), h3)   # Compute the update for output layer
    output_weights.assign(output_weights + tf.transpose(delta_output_weights))  # Update the output layer weights




























# Instantiate model
dfa_model = DFAModel()

# Forward pass with dummy data to build the model
dummy_input = tf.random.normal((1, dataset.shape[1]))  # Match the shape of your input data
_ = dfa_model(dummy_input, no_feedback=True)  # Perform a forward pass to build the model
dfa_model.summary()


# Training loop
epochs = 40
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
        # custom_backprop(dfa_model, x_batch, y_batch, learning_rate) # <= This works well






# Evaluate the model
predictions = dfa_model(dataset, no_feedback=True)
print(predictions)
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



test_predictions = dfa_model(testset, no_feedback=True)

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


