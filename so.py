import numpy 
import random 

class Trainer():
    #Передать нейросеть и data в виде [ [[1, 0, 0], [1]] ] 
    def __init__(self, network, data: list, epohs: int = 100, generation_size: int = 10, learning_rate: float = 1):
        self.networks = []
        self.best_network = network

        self.epohs = epohs
        self.learning_rate = learning_rate
        self.generation_size = generation_size 

        self.dataset = data
    
    #Обучаем нейросети
    def train(self):
        for i in range(self.epohs):
            self.create_generation(self.best_network, self.generation_size, self.learning_rate)
            self.feed_forward_networks()

            #Посчитать общую ошибку
            for n in self.networks: n.calcLoss()

            #Отсортировать и выбирать лучшего
            self.networks = sorted(self.networks, key=lambda net: net.loss)
            self.best_network = self.networks[0]

            #Очистим информацию о ошибках
            self.best_network.clear_results()

        return self.best_network

    #Сгенерировать новые версии нейросети
    def create_generation(self, network, generation_size: int, l_r: float): 
        self.networks = []
        self.networks.append(network) #Добавить лучшую для стабилизации

        #На 1 меньше ибо дбавлена изначальная 
        for i in range(generation_size - 1):
            new_network = network.clone()
            new_network.change_weights(l_r)
            self.networks.append(new_network)
    
    #Прогнать данные через нейросети
    def feed_forward_networks(self):
        for n in self.networks:
            for i in range(len(self.dataset)):
                n.feed_forward(self.dataset[i][0])
                n.calc_local_loss(self.dataset[i][1])