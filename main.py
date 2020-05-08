'''
Created on 5 mai 2020

@author: George
'''

from transform import loadPokemons, liniarize_image, toSepia, loadImagesSimple,\
    processImage
import cv2
import numpy as np
from net import load
from net2 import loadNet

if __name__ == '__main__':
    
#    import csv_loader
    import net as network
    #training_data, validation_data, test_data = csv_loader.load_data()
    #training_data, validation_data, testing_data = loadPokemons()
    training_data, validation_data, testing_data = loadImagesSimple()
    validation_data = list(validation_data)
    training_data = list(training_data)
    
    #net = network.Network([100, 10, 2], cost=network.QuadraticCost)
    net = network.Network([100, 10, 2], cost=network.QuadraticCost)
    #net.SGD(training_data, 10, 10, 0.2, lmbda=1.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)
    net.SGD(training_data, 50, 10, 0.1, 0.0, evaluation_data=list(validation_data), monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
    net.save("DANN_distance.txt")
    
    # best result so far in DANN_100.txt
    
    #net = load("DANN_distance.txt")
    a = toSepia(cv2.imread("resized/images109.png"))
    a = processImage(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))
    
    a = toSepia(cv2.imread("resized/images109_sepia.png"))
    a = processImage(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))
    
    '''training_data, validation_data, testing_data = loadImagesSimple()
    validation_data = list(validation_data)
    training_data = list(training_data)
    filename = 'net2.txt'
    import net2 as network
    net = network.Network([300,50,2])
    net.SGD(training_data, 100, 10, 0.1, testing_data)
    net.save(filename)'''
    
    '''net = loadNet('net2.txt')
    
    a = toSepia(cv2.imread("resized/images109.png"))
    a = liniarize_image(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))
    
    a = toSepia(cv2.imread("resized/images109_sepia.png"))
    a = liniarize_image(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))
    
    
    a = toSepia(cv2.imread("resized/images9.png"))
    a = liniarize_image(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))
    
    a = toSepia(cv2.imread("resized/images9_sepia.png"))
    a = liniarize_image(a)
    print(net.feedforward(a))
    print(np.argmax(net.feedforward(a)))'''
    
