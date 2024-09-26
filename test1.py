import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

# Dataset generation

def CreateSpiral(n_points=1000, start_theta=0, expansion_rate=0.2, turns=2):
    
    theta = np.linspace(0, turns * 2*np.pi, n_points)  

    # Generate r values (radius)
    r = expansion_rate * theta

    # Convert to Cartesian coordinates
    x = r * np.cos(theta+start_theta)
    y = r * np.sin(theta+start_theta)

    return x,y


n_points=5000

x1,y1 = CreateSpiral(n_points)

x2,y2 = CreateSpiral(n_points, start_theta=np.pi)

if True:
    plt.scatter(x1,y1, label="spiral 1")
    plt.scatter(x2,y2, label="spiral 2")
    plt.legend()
    plt.show()

# Full dataset :
x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

print(len(x1), len(x2), len(y1), len(y2), len(x), len(y))
dataset = np.array([ np.array([x[i], y[i]]) for i in range(len(x))   ])


expected_results = np.array(
                                [ np.array([1,0]) for i in range(n_points)] + 
                                [ np.array([0,1]) for i in range(n_points)] 
                            )

# The dataset is a numpy array like so [ [x1,y1], [x2,y2] , ... , [xn,yn]]
# The expected_results is a numpy array with the expected classification result ([0,1] or [1,0])



# Model



# Define the model
model = Sequential([
    layers.Input(shape=(2,)),             # Input layer with 2 neurons
    layers.Dense(100, activation='relu'),   # First hidden layer with 32 neurons
    layers.Dense(100, activation='relu'),   # Second hidden layer with 32 neurons
    layers.Dense(100, activation='relu'),   # Third hidden layer with 32 neurons
    layers.Dense(2)                        # Output layer with 2 neurons
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error', metrics=['accuracy'])
model.fit(dataset, expected_results, epochs=400, batch_size=200, shuffle=True) #using default backpropagation

# Evaluate the model
loss, accuracy = model.evaluate(dataset, expected_results)
print(f"Loss: {loss}, Accuracy: {accuracy}")








# Introducing noise lol
noise = 0.22


# Test our network on sample dataset
xt1,yt1 = CreateSpiral(200)
xt2,yt2 = CreateSpiral(200, start_theta=np.pi)

# Full testset :
xt = np.concatenate((xt1, xt2))
yt = np.concatenate((yt1, yt2))

testset = np.array([ np.array([xt[i]+np.random.rand()*noise, yt[i]+np.random.rand()*noise]) for i in range(len(xt))   ])


predictions = model.predict(testset)

# print( predictions)

spiral1 = []
spiral2 = []
spiralUnk = []

for i in range(len(predictions)):
    
    pred = predictions[i]

    if pred[0]>0.5 and pred[1]<0.5:
        spiral1.append(testset[i])
    
    elif pred[0]<0.5 and pred[1]>0.5:
        spiral2.append(testset[i])

    else:
        spiralUnk.append(testset[i])



def PlotSpiral(data, label, style="bo"):
    x = [dat[0] for dat in data]
    y = [dat[1] for dat in data]
    plt.plot(x,y, style, label=label)

PlotSpiral(spiral1, "Spiral 1", "b.")
PlotSpiral(spiral2, "Spiral 2", "r.")
PlotSpiral(spiralUnk, "Unknown", "g.")

plt.legend()
plt.show()


model.summary()
