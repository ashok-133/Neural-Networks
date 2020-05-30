import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def sigmoid_derivative(x):
    return x*(1-x)


train_input = np.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]])

train_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

weights = 2*np.random.random((3,1)) - 1

print(weights)
for iteration in range(20000):
    input_layer = train_input
    outputs = sigmoid(np.dot(input_layer,weights))
    error = train_outputs - outputs
    adjustments = error*sigmoid_derivative(outputs)
    weights +=np.dot(input_layer.T,adjustments)
print(outputs)