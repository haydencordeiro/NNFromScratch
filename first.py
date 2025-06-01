import math
from random import random
import matplotlib.pyplot as plt

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

COLORS = {
    'background': '#121212',              # --clr-surface-a0
    'text': '#ffffff',                    # --clr-light-a0
    'circle_edge': '#ffffff',
    'circle_fill': '#3f3f3f',             # --clr-surface-a20
    'error_text': '#ff6b6b',
    'output_text': '#ffffff',
    'weight_text': '#aaa3eb',             # --clr-primary-a50
    'edge': '#7c78df',                    # --clr-primary-a30
}

def visualize_network(network, input_len, epoch, loss, ax=None):
    """
    Dark-mode network visualizer with input layer support.

    Args:
        network (list): Your NETWORK object
        input_len (int): Number of input features (neurons in input layer)
        epoch (int): Current epoch
        loss (float): Current epoch loss (SSE)
        ax (matplotlib axis): Optional axis
    """
    layers = len(network)
    max_neurons = max(input_len, *(len(l) for l in network))
    col_spacing, row_spacing = 2.5, 1.5

    fig, ax = plt.subplots(figsize=(col_spacing*(layers+1)+2, row_spacing*max_neurons+2))
    ax.set_facecolor(COLORS['background'])
    ax.axis("off")

    # Title / Loss display
    ax.text(0.01, 0.99, f"Epoch {epoch}", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, weight="bold", color=COLORS['text'])
    ax.text(0.99, 0.99, f"SSE={loss:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color=COLORS['text'],
            bbox=dict(boxstyle="round", fc=COLORS['circle_fill'], ec="none"))

    positions = {}

    # -- Input layer (reconstructed from input_len) --
    x = 0
    offset = (max_neurons - input_len) / 2.0
    for ni in range(input_len):
        y = (offset + ni) * row_spacing
        positions[(0, ni)] = (x, -y)
        circ = plt.Circle((x, -y), 0.3, color=COLORS['background'], ec=COLORS['circle_edge'], lw=1.2)
        ax.add_patch(circ)
        ax.text(x, -y, f"x{ni}", ha="center", va="center", fontsize=8, color=COLORS['text'])

    # -- Draw hidden/output layers and weights --
    for li, layer in enumerate(network):
        x = (li + 1) * col_spacing
        offset = (max_neurons - len(layer)) / 2.0
        for ni, neuron in enumerate(layer):
            y = (offset + ni) * row_spacing
            positions[(li + 1, ni)] = (x, -y)

            # Draw neuron circle
            circ = plt.Circle((x, -y), 0.3, color=COLORS['circle_fill'], ec=COLORS['circle_edge'])
            ax.add_patch(circ)

            # Output value inside circle
            ax.text(x, -y, f"{neuron['output']:.2f}", ha="center", va="center", fontsize=8, color=COLORS['output_text'])
            # Error (δ) below circle
            ax.text(x, -y-0.4, f"δ={neuron['error']:.2f}", ha="center", va="center", fontsize=6, color=COLORS['error_text'])

            # Draw incoming connections
            prev_layer_idx = li
            for pj, w in enumerate(neuron['incoming_weights']):
                prev_x, prev_y = positions[(prev_layer_idx, pj)]
                ax.plot([prev_x, x], [prev_y, -y], lw=0.6, color=COLORS['edge'])
                mid_x = prev_x + 0.33 * (x - prev_x)
                mid_y = prev_y + 0.33 * (-y - prev_y)
                ax.text(mid_x, mid_y, f"{w:.2f}", fontsize=6, color=COLORS['weight_text'], ha="center", va="center")

    ax.set_xlim(-1, (layers + 1) * col_spacing)
    ax.set_ylim(-max_neurons * row_spacing - 1, 1)
    plt.tight_layout()
    plt.savefig(f'frames/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # plt.show()


input_dataset = [(0, 0), (0, 1), (1, 0), (1, 1)]
# one hot encoding the output for each input
target_outputs = [(0, 1), (1, 0), (1, 0), (0, 1)]

# Initializing the network with 2 input neurons, 3 hidden neurons, and 2 output neurons
create_network([2,3,2])

# Training the network
for i in range(200):
    errorDuringEpoch = 0
    for idx,input_data in enumerate(input_dataset):
        outputs = forward_pass(list(input_data))
        errorDuringEpoch += sum((target_outputs[idx][j]- outputs[j])**2 for j in range(len(outputs)))
        # print(f"Input: {input_data}")
        # display_network()
        backpropagation(target_outputs[idx])
        update_weights(list(input_data), learning_rate=0.5)
    print(f"Epoch {i+1}, Error: {errorDuringEpoch}")
    visualize_network(NETWORK, 2, i+1, errorDuringEpoch)

print(forward_pass((0, 0)))
print(forward_pass((0, 1)))
print(forward_pass((1, 0)))
print(forward_pass((1, 1)))
# display_network()
