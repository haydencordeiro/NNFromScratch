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


def activation_function(x):
    return 1.0/(1.0 + math.exp(-x))  # Sigmoid activation function

def forward_pass(input_data):
    global NETWORK
    # For each layer except the last one, calculate the output for each neuron
    for layer_idx in range(len(NETWORK)):
        current_layer = NETWORK[layer_idx]
        new_input_data = []
        print(f"Processing Layer {layer_idx + 1}", input_data)
        for neuron_idx, neuron in enumerate(current_layer):
            # Calculate the weighted sum of inputs
            weighted_sum = neuron['bias']
            for weight_idx,weight in enumerate(neuron['incoming_weights']):
                weighted_sum += weight * input_data[weight_idx]
            # Apply the activation function
            neuron['output'] = activation_function(weighted_sum)
            new_input_data.append(neuron['output'])
        input_data = new_input_data




input_dataset = [(0, 0), (0, 1), (1, 0), (1, 1)]
# one hot encoding the output for each input
target_outputs = [(0, 1), (1, 0), (1, 0), (0, 1)]

# Initializing the network with 2 input neurons, 3 hidden neurons, and 2 output neurons
create_network([2,3,2])

for input_data in input_dataset:
    forward_pass(list(input_data))
    print(f"Input: {input_data}")
    display_network()

