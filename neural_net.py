import matplotlib.pyplot as plt
import numpy as np
import struct
from tqdm import tqdm


# defining the activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# defining the derivative of the activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# applied to output layer
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability fix
    return exps / np.sum(exps, axis=1, keepdims=True)

# defining the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize weights randomly and set biases to zero
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / (input_size + hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / (hidden_size + output_size))
        self.b2 = np.zeros((1, output_size))

    # defining the forward pass
    def forward(self, x):
        # store the input matrix for backpropogation
        self.x = x
        # move through first layer and apply activation function
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        # move through second layer and apply activation function for final output
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)

        return self.a2

    # computes the loss of a prediction 
    def compute_loss(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # bounds y_pred so that it fits within the domain of log
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        return loss

    # computes the gradients for each weight and bias
    def back_prop(self, x, y_true):
        # store the batch size
        n = x.shape[0]
        # find the output error
        delta2 = self.a2 - y_true
        # find product of partial derivatives
        delta1 = np.dot((delta2), self.w2.T) * sigmoid_derivative(self.a1) 
        # dot the product with our transposed input matrix
        self.dw1 = np.dot(x.T, delta1) / n
        self.db1 = np.sum(delta1, axis = 0, keepdims=True) / n
        self.dw2 = np.dot(self.a1.T, (delta2)) / n
        self.db2 = np.sum(delta2, axis = 0, keepdims=True) / n

    # moves each weight and bias in accordance to its partial derivative
    def update_parameters(self, learning_rate):
        # learning rate
        n = learning_rate
        # update weights and biases
        self.w1 = self.w1 - n * self.dw1
        self.b1 = self.b1 - n * self.db1
        self.w2 = self.w2 - n * self.dw2
        self.b2 = self.b2 - n * self.db2

    # loads the images from the MNIST files
    def load_images(self, filename):
        # read and parse through the header data
        with open(filename, 'rb') as f:
            header_bytes = f.read(16)
            magic, num_images, rows, cols = struct.unpack(">IIII", header_bytes)
            # read and organize the image data
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)
            images = images.reshape(num_images, rows * cols) / 255.0
        return images

    # loads the labels of the images in the MIST files
    def load_labels(self, filename):
        # read and parse through header data
        with open(filename, 'rb') as f:
            header_bytes = f.read(8)
            magic, num_labels = struct.unpack(">II", header_bytes)
            # read and organize label data
            label_data = f.read()
            label_data = np.frombuffer(label_data, dtype=np.uint8)
        return label_data
    
    # formats into one-hot encoded
    def one_hot_encode(self, y):
        one_hot = np.zeros((y.size, 10))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    
    # trains the neural network
    def train(model, x_train, y_train, x_test, y_test, epochs, batch_size, learning_rate):
        for epoch in tqdm(range(epochs)): # loops through number of epochs 
            #shuffles the training data
            perm = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm]
            y_train = y_train[perm]
            epoch_loss = 0
            num_batches = 0
            for i in range(0, x_train.shape[0], batch_size): # increments by batch size
                x_batch = x_train[i:i+batch_size] # gets image data for the batch
                y_batch = y_train[i:i+batch_size] # gets label data for the batch
                y_batch = model.one_hot_encode(y_batch)

                # runs the forward pass on our batch
                y_pred = model.forward(x_batch)

                # runs the backpropogation on our batch
                model.back_prop(x_batch, y_batch)

                # computes the loss of this epoch to track progress
                loss = model.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                num_batches += 1

                # adjust weights and biases for learning
                model.update_parameters(learning_rate)
            print(epoch_loss / num_batches)
            # test accuracy on test set
            model.test(x_test, y_test)

    # used to test the accuracy
    def test(model, x_test, y_test):
        correct = 0
        for i in range(x_test.shape[0]):
            output = np.argmax(model.forward(x_test[i]))
            if (y_test[i] == output):
                correct += 1
        print(str(correct) + " correctly identified out of " + str(x_test.shape[0]) + " images")
