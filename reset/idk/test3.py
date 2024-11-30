import tensorflow as tf
import numpy as np
from idk.dfa import direct_feedback_alignement




class SimpleANN(tf.keras.Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleANN, self).__init__()
        self.hidden_layers = []
        prev_size = input_size

        # Define hidden layers
        for hidden_size in hidden_sizes:
            layer = tf.keras.layers.Dense(hidden_size, activation='relu')
            self.hidden_layers.append(layer)
            prev_size = hidden_size

        # Define output layer
        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def direct_feedback_alignment(model, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, outputs))

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Sample usage
# Replace with your actual dataset
input_size = 784  # e.g., for MNIST
hidden_sizes = [128, 64]  # Example hidden layer sizes
output_size = 10  # Example number of classes

model = SimpleANN(input_size, hidden_sizes, output_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Sample data - replace with your data
train_data = np.random.rand(1000, 784).astype(np.float32)  # 1000 samples of 784 features
train_labels = np.random.randint(0, 2, (1000, 10)).astype(np.float32)  # 1000 samples of 10 classes (one-hot encoded)

num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    for batch in range(0, len(train_data), batch_size):
        batch_data = train_data[batch:batch + batch_size]
        batch_labels = train_labels[batch:batch + batch_size]
        loss_value = direct_feedback_alignment(model, optimizer, batch_data, batch_labels)

    print("Epoch:", epoch, "Loss:", loss_value)
