'''
Created on 27 sept. 2019

@author: George
'''



import numpy as np
from transform import writeBitmapToTempImage, readIntoRGBGrayscale, liniarize_image, readIntoSepia



def vectorized_result(j):
    # sepia and non-sepia
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def load_data():
    #f = open('test.txt', 'r')
    f = open('emnist-balanced-train.csv', 'r')
    training_inputs = []
    training_results = []
    
    splitter = 0
    
    while True:
        line = f.readline()
        parts = line.strip().split(",")
        if len(parts) == 1:
            break
        code = None
        stringArray = parts[1:]
        hexArray = [int(x) for x in stringArray]
        
        bitmap = hexArray
        writeBitmapToTempImage(bitmap)
        if splitter % 2 :
            img = readIntoRGBGrayscale()
            img = liniarize_image(img)
            code = 0 # normal images
        else:
            img = readIntoSepia()
            img = liniarize_image(img)
            code = 1 # sepia images
        #training_input = np.reshape(bitArray, (2352, 1)) # 3 channels X 784 = 28 X 28 images
        img = np.reshape(img, (300, 1))
        #img = np.reshape(img, (784, 3))
        splitter += 1
        training_input = img
        training_inputs.append(training_input)
        training_results.append(vectorized_result(code))
    training_data = zip(training_inputs, training_results)
    f.close()
    
    #g = open('test.txt', 'r')
    g = open('emnist-balanced-test.csv', 'r')
    testing_inputs = []
    testing_results = []
    while True:
        line = g.readline()
        parts = line.strip().split(",")
        if len(parts) == 1:
            break
        code = int(parts[0])
        stringArray = parts[1:]
        hexArray = [int(x) for x in stringArray]
        
        bitmap = hexArray
        writeBitmapToTempImage(bitmap)
        if splitter % 2 :
            img = readIntoRGBGrayscale()
            img = liniarize_image(img)
            code = 0
        else:
            img = readIntoSepia()
            img = liniarize_image(img)
            code = 1
        #training_input = np.reshape(bitArray, (2352, 1)) # 3 channels X 784 = 28 X 28 images
        img = np.reshape(img, (300, 1))
        #img = np.reshape(img, (784, 3))
        splitter += 1
        testing_input = img

        testing_inputs.append(testing_input)
        testing_results.append(vectorized_result(code))
    testing_data = zip(testing_inputs, testing_results)
    g.close()
    
    validation_data = []
    for tup in testing_data:
        vectorized_rez = tup[1]
        rez = (np.argmax(vectorized_rez))
        validation_data.append((tup[0],np.int64(rez)))

    return training_data, validation_data, testing_data
    #return training_data, validation_data, training_data

