import json

#Передать network.weights и network.biases
def save_network(weights: list, biases: list, filename: str = "network.json"):
    network = encode_json(weights, biases)
    with open(filename, "w") as f:
        json.dump(network, f, indent=4)

def load_network(filename: str = "network.json"):
    with open(filename, "r") as f:
        network = json.load(f)

    weights = []
    biases = []

    for layer in network["layers"]:
        layer_weights = []
        layer_biases = []
        for neuron in layer["neurons"]:
            layer_weights.append(neuron["weights"])
            layer_biases.append(neuron["bias"])
        weights.append(layer_weights)
        biases.append(layer_biases)

    return weights, biases

def encode_json(w: list, b: list):
    network = {"layers": []}

    for layer in range(len(w)):
        layer_dict = {"neurons": []}

        for neuron in range(len(w[layer])):
            neuron_dict = {"weights": w[layer][neuron].tolist(), 
                           "bias": float(b[layer][neuron])}
            layer_dict["neurons"].append(neuron_dict)

        network["layers"].append(layer_dict)
    
    return network