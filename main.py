from network.ns0_1 import Network as nn
import trainingModule.so as so 
import saveAndLoad.storage_json as sj 
from time import perf_counter

data = [
    [
        [1, 0, 0, 0], [1]
    ],
    [
        [0, 1, 1, 0], [0]
    ],
    [
        [0, 0, 0, 1], [0]
    ],
    [
        [1, 0, 0, 1], [1]
    ]
]

test_data = [    
    [
        [1, 1, 0, 0], [1]
    ],
    [
        [1, 0, 1, 0], [1]
    ],
    [
        [0, 0, 0, 0], [0]
    ]
]

network = nn([4, 5, 1], clip_number = 5)
trainer = so.Trainer(network, data, 10000, 2, 0.1)

network = trainer.train()

for i in range(len(test_data)):
    result = network.feed_forward(test_data[i][0])
    print(f"{result} / {test_data[i][1]}")

sj.save_network(network.weights, network.biases)