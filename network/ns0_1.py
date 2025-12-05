import numpy as np
import copy
import random 

class Network():
    def __init__(self, network_structure: list = None, clip_number: float = 25, weights: list = None, biases: list = None):
        #Нужно для ограничения весов
        self.clip_number = clip_number

        self.loss: float = 0
        self.results: list = []

        self.weights: list = []
        self.biases: list = []
        self.outputs: list = []

        #Если переданы веса, загрузить. 
        if weights is not None and biases is not None:
            self.weights = [np.array(w).copy() for w in weights]
            self.biases = [np.array(b).copy() for b in biases]
        #Если передана структура создать веса
        elif network_structure is not None:
            for i in range(1, len(network_structure)):
                layer_weights = np.random.randn(network_structure[i], network_structure[i - 1])
                layer_biases = np.random.randn(network_structure[i])

                self.weights.append(layer_weights)
                self.biases.append(layer_biases)
        #Если не передано ничего
        else:
            print("ОШИБКА: Передайте веса и смещения или структуру сети!")

    #Передать data одинарным списком - [1, 2, 3, 4]
    def feed_forward(self, data: list):
        #Добавить входные данные для активации
        self.outputs: list = []
        self.outputs.append(np.array(data))

        #Перемножить все слои кроме выходного
        for layer in range(len(self.weights) - 1):
            output = np.dot(self.weights[layer], self.outputs[layer]) + self.biases[layer]
            output = relu(output)
            self.outputs.append(output)
        
        #Перемножить веса выходного слоя
        output = np.dot(self.weights[-1], self.outputs[-1]) + self.biases[-1]
        output = sigmoid(output)
        self.outputs.append(output)

        return self.outputs[-1]
    
    #Поменять веса
    def change_weights(self, l_r: float):
        for layer in range(len(self.weights)):
            # Мутация весов
            np.add(self.weights[layer], 
                   np.random.uniform(-l_r, l_r, size = self.weights[layer].shape), 
                   out = self.weights[layer])

            # Мутация смещений
            np.add(self.biases[layer], 
                   np.random.uniform(-l_r, l_r, size = self.biases[layer].shape), 
                   out = self.biases[layer])
            
            #Ограничим веса чтобы не разлетались
            np.clip(self.weights[layer], -self.clip_number, self.clip_number, out=self.weights[layer])
            np.clip(self.biases[layer], -self.clip_number, self.clip_number, out=self.biases[layer])
    
    #Копирование нейронной сети
    def clone(self):
        new_network = Network.__new__(Network)
        new_network.clip_number = self.clip_number
        new_network.weights = [w.copy() for w in self.weights]
        new_network.biases = [b.copy() for b in self.biases]
        new_network.results = []
        new_network.outputs = []
        new_network.loss = 0
        return new_network

    #Посчитать loss для одной задачи. Вызывать после каждой задачи.
    def calc_local_loss(self, target):
        answer = self.outputs[-1]
        
        res = np.sum((target - answer) ** 2)
        self.results.append(res)
    
    #Посчитать loss для батча. Вызывать после каждого батча. 
    def calcLoss(self):
        self.loss = np.mean(self.results)
    
    #Очистить нейронную сеть от записей ошибок. ОБЯЗАТЕЛЬНО ВЫЗЫВАТЬ ПОСЛЕ КАЖДОГО БАТЧА!
    def clear_results(self):
        self.results = []
        self.loss = 0

#Функции активации
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
