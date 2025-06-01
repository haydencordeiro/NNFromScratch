
from random import random
NETWORK = []

#  layer is list, the len of the list denotes the number of  layers and each element determines the number of nuerons in that  layer
def create_network( layers):
    global NETWORK
    NETWORK = []

    # need to zip the  layers so that we can get the number of neurons in the current  layer and the next  layer
    layer_to_next_list = list(zip(layers[:-1], layers[1:]))
    # the last  layer will not have a next layer, so we will set it to 0
    layer_to_next_list.append((layers[-1], 0))
    for no_of_neurons_in_current__layer,no_of_neurons_in_next__layer  in layer_to_next_list:
        layer =[]
        for i in range(no_of_neurons_in_current__layer ):
            layer.append({
                'outgoing_weights' : [random() for _ in range(no_of_neurons_in_next__layer)],
                'output' : 0,
                'bias' : random(),
            })
        NETWORK.append(layer)



# Initializing the network with 2 input neurons, 3 hidden neurons, and 2 output neurons
create_network([2,3,2])

# Displaying the structure of the network
for idx,layer in enumerate(NETWORK):
    print(f"Layer {idx+1}:")
    for neuron_idx, neuron in enumerate(layer):
        print(f"  Neuron {neuron_idx+1}:")
        print(f"    Bias: {neuron['bias']}")
        print(f"    Outgoing Weights: {neuron['outgoing_weights']}")
        print(f"    Output: {neuron['output']}")

