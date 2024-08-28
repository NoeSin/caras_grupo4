import numpy as np
import pandas as pd
import tensorflow as tf


# disable scientific notation
np.set_printoptions(suppress=True, precision=2)

## Creamos clases para las capas, 
# cada capa sabe comoa realizr el forward y backward de la red

# layer class
class Layer:
    def __init__(self, n_neurons, n_input):
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.weights = np.random.uniform( -0.5, 0.5, size=(n_neurons, n_input) )  # inicializamos con pesos random
        self.biases = np.random.uniform(-0.5, 0.5, size=(n_neurons, 1))  # inicializamos sesgos random (w0)

    # forward
    def forward(self, inputs):
        self.inputs = inputs  # guardamos los impust recibidos para realizar el back
        self.outputs = (  np.dot(self.weights, self.inputs) + self.biases  )  # calculamos la salida, con mult de matriz + sesgos(w0)
        return self.outputs

    # backward prop
    def backward(self, output_gradient, learning_rate):
        prev_layer_gradient = np.dot( self.weights.T, output_gradient )  # calculamos e gradiente para la capa previa
        weights_gradient = np.dot( output_gradient, self.inputs.T )  # calculamos el gradiente para los pesos 
        self.weights -= learning_rate * weights_gradient  # actualizamos pesos
        self.biases -= learning_rate * output_gradient  # actualizamos biases
        return prev_layer_gradient


# activation classes
# linear
class Purelin:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, output_gradient, learning_rate):
        return output_gradient # Para una función lineal, la derivada es simplemente el gradiente de salida, ya que la función es lineal.


# sigmoid
class Logsig:
    def __init__(self):
        self.output = 0

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient, learning_rate):
      
        return np.multiply(output_gradient, self.output * (1 - self.output)) #La derivada de la función sigmoide es y * (1 - y), donde y es la salida de la función sigmoide


# tanh
class Tansig:
    def __init__(self):
        self.output = 0

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - self.output * self.output) #La derivada de tanh(y) es 1 - tanh(y)^2.



class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        x -= np.max(x, axis=1, keepdims=True)  # Softmax stabilization trick
        exps = np.exp(x)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        n = self.output.shape[1]
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i, j] = self.output[0, i] * (1 - self.output[0, i])
                else:
                    jacobian[i, j] = -self.output[0, i] * self.output[0, j]
        return np.dot(jacobian, output_gradient)
    
    
    
    
# loss MSE


def MSE_loss(y_true, y_pred):
    loss = np.mean(np.power(y_true - y_pred, 2))
    return loss


def MSE_loss_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


# loss cross entropy


def cross_loss(y_true, y_pred):
    limit = 1e-10
    y_pred = np.clip(
        y_pred, limit, 1 - limit
    )  # prevents y_pred=0 since log(0) is undefined
    losses = -np.sum(y_true * np.log(y_pred), axis=0)
    return np.mean(losses)


def cross_loss_prime(y_true, y_pred):
    size = y_true.shape[0]
    # return (y_pred - y_true) / size
    return -(y_true / y_pred) / size


class CrossEntropyLoss:
    def __init__(self):
        self.limit = 1e-10

    def calc(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.limit, 1 - self.limit)
        losses = -np.sum(y_true * np.log(y_pred), axis=0)
        return np.mean(losses)

    def prime(self, y_true, y_pred):
        epsilon = 1e-10
        size = y_true.shape[0]
        return -(y_true / (y_pred + epsilon)) / size
    
    
#    import numpy as np



class n_network:
    def __init__(self, params, layers, activations, loss_function):
        self.layers = layers  # list of layer clasess
        self.learning_rate = params["learning_rate"]
        self.max_epochs = params["epochs"]
        self.target_error = params["target_error"]
        self.weights = []
        self.biases = []
        self.activations = activations  # list of activations per layer
        self.loss = loss_function  # CrossEntropyLoss class in functions_nn.py
        self.accuracy = 0
        self.trained = False

    def train(self, X, Y):
        epoch = 0
        error = 10

        while error > self.target_error and epoch < self.max_epochs:
            error = 0
            for x, y in zip(X, Y):

                # Forward pass
                activation = self.single_forward(x)

                # Calculate loss and gradient
                error += self.loss.calc(y, activation)
                gradient = self.loss.prime(y, activation)

                # Backward pass
                layers_and_activs = [
                    val for pair in zip(self.layers, self.activations) for val in pair
                ]  
                for layer in reversed(layers_and_activs):
                    gradient = layer.backward(gradient, self.learning_rate)

            # Update loop params
            epoch += 1
            error /= len(X)

            # Save parameters
            self.save_params()
            
            # print epoch error
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} - Error: {error}")

            # adaptive learning rate
            if epoch % 2000 == 0:
                self.learning_rate /= 10
        # Update trained flag
        self.trained = True

    def single_forward(self, x):  # Forward pass of a single observation
        # input
        activation = x
        # hidden + output layers
        for layer, activ_fun in zip(self.layers, self.activations):
            output = layer.forward(activation)  # z_j
            activation = activ_fun.forward(output)  # a_j
        return activation

    def predict(self, X):
        if not self.trained:
            print("El modelo no está entrenado")
            return
        activations = [self.single_forward(x) for x in X]
        predictions = [np.argmax(activ, axis=0)[0] for activ in activations]
        return predictions

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        correct_preds = [pred == np.argmax(y) for pred, y in zip(predictions, Y)]
        self.accuracy = np.mean(correct_preds)
        print(f"Accuracy: {self.accuracy}")
        return self.accuracy

    def save_params(self):
        self.weights += [layer.weights for layer in self.layers]
        self.biases += [layer.biases for layer in self.layers]
        
    def load_params(self):
    	for i, layer in enumerate(self.layers):
            layer.weights = pd.read_csv(f"3.nn_params/weights_{i}.csv", header=None).to_numpy()
            layer.biases = pd.read_csv(f"3.nn_params/biases_{i}.csv", header=None).to_numpy()
            self.trained = True

    def __str__(self) -> str:
            result = f"Neural Network with {len(self.layers)} layers:\n"
            for i, layer in enumerate(self.layers):
                result += f"Layer {i+1}:\n"
                result += f"Weights: {layer.weights}\n"
                result += f"Biases: {layer.biases}\n"
            return result

    def export_params(self):
            for i, layer in enumerate(self.layers):
                np.savetxt(f"3.nn_params/weights_{i}.csv", layer.weights, delimiter=",")
                np.savetxt(f"3.nn_params/biases_{i}.csv", layer.biases, delimiter=",")
            print("Parameters exported")