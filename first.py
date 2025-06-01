import math
from random import random
NETWORK = []

#  layer is list, the len of the list denotes the number of  layers and each element determines the number of nuerons in that  layer
# Note Layer 0 is the input layer, Layer 1 is the first hidden layer, and so on.
def create_network( layers):
    global NETWORK
    NETWORK = []

    # the first layer will not have any incoming weights so there will be no need for backpropagation for the first layer so we will skip it
    # need to zip the  layers so that we can get the number of neurons in the prev layer and the current  layer
    layer_to_next_list = list(zip(layers[:-1], layers[1:]))
    print(layer_to_next_list)
    for no_of_neurons_in_prev_layer,no_of_neurons_in_current__layer  in layer_to_next_list:
        layer =[]
        for i in range(no_of_neurons_in_current__layer ):
            layer.append({
                'incoming_weights' : [random() for _ in range(no_of_neurons_in_prev_layer)],
                'output' : 0,
                'error' : 0,
                'bias' : random(),
            })
        NETWORK.append(layer)


def display_network():
    global NETWORK
    for idx, layer in enumerate(NETWORK):
        print(f"Layer {idx+1}:")
        for neuron_idx, neuron in enumerate(layer):
            print(f"  Neuron {neuron_idx+1}:")
            print(f"    Bias: {neuron['bias']}")
            print(f"    Incoming Weights: {neuron['incoming_weights']}")
            print(f"    Output: {neuron['output']}")
            print(f"    Error: {neuron['error']}")


def activation_function(x):
    return 1.0/(1.0 + math.exp(-x))  # Sigmoid activation function

def forward_pass(input_data):
    global NETWORK
    # For each layer except the last one, calculate the output for each neuron
    for layer_idx in range(len(NETWORK)):
        current_layer = NETWORK[layer_idx]
        new_input_data = []
        # print(f"Processing Layer {layer_idx + 1}", input_data)
        for neuron_idx, neuron in enumerate(current_layer):
            # Calculate the weighted sum of inputs
            weighted_sum = neuron['bias']
            for weight_idx,weight in enumerate(neuron['incoming_weights']):
                weighted_sum += weight * input_data[weight_idx]
            # Apply the activation function
            neuron['output'] = activation_function(weighted_sum)
            new_input_data.append(neuron['output'])
        input_data = new_input_data
    return input_data  # Return the output of the last layer

def sigmoid_derivative(output):
    return output * (1 - output)  # Derivative of the sigmoid function

# This will help use calculate error that will be used to adjust the weights
def backpropagation(target_outputs):
    global NETWORK
    # loop through the network in reverse order
    for layer_idx in reversed(range(len(NETWORK))):
        current_layer = NETWORK[layer_idx]
        next_layer = NETWORK[layer_idx + 1] if layer_idx + 1 < len(NETWORK) else None
        
        # If it's the output layer, calculate the error
        if next_layer is None:
            for neuron_idx, neuron in enumerate(current_layer):
                output = neuron['output']
                target = target_outputs[neuron_idx]
                # Calculate the error for the output neuron
                error = (output-target) * sigmoid_derivative(output)
                neuron['error'] = error 
        else:
            # For hidden layers, propagate the error back to adjust weights
            for neuron_idx, neuron in enumerate(current_layer):
                error = 0
                for next_neuron in next_layer:
                    error += next_neuron['incoming_weights'][neuron_idx] * next_neuron['error']
                error *= sigmoid_derivative(neuron['output'])
                neuron['error'] = error

def update_weights(input_data, learning_rate=0.1):
    global NETWORK
    # Update weights and biases for each neuron
    for layer_idx in range(len(NETWORK)):
        current_layer = NETWORK[layer_idx]
        if layer_idx != 0:
            # For the first hidden layer we use the actauly input data
            # otherwise we use the output of the previous layer
            input_data = [neuron['output'] for neuron in NETWORK[layer_idx - 1]]

        for neuron in current_layer:
            # Update incoming weights
            for weight_idx in range(len(neuron['incoming_weights'])):
                neuron['incoming_weights'][weight_idx] -= learning_rate * neuron['error'] * input_data[weight_idx]
            # Update bias
            neuron['bias'] -= learning_rate * neuron['error']


input_dataset = [(0, 0), (0, 1), (1, 0), (1, 1)]
# one hot encoding the output for each input
target_outputs = [(0, 1), (1, 0), (1, 0), (0, 1)]

# Initializing the network with 2 input neurons, 3 hidden neurons, and 2 output neurons
create_network([2,3,2])

# Training the network
for i in range(100):
    errorDuringEpoch = 0
    for idx,input_data in enumerate(input_dataset):
        outputs = forward_pass(list(input_data))
        errorDuringEpoch += sum((target_outputs[idx][j]- outputs[j])**2 for j in range(len(outputs)))
        # print(f"Input: {input_data}")
        # display_network()
        backpropagation(target_outputs[idx])
        update_weights(list(input_data), learning_rate=0.1)
    print(f"Epoch {i+1}, Error: {errorDuringEpoch}")

print(forward_pass((0, 0)))
print(forward_pass((0, 1)))
print(forward_pass((1, 0)))
print(forward_pass((1, 1)))
# display_network()
