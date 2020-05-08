'''
Created on 5 mai 2020

@author: George
'''


import cv2
import numpy as np
from copy import copy



our_secret_bitmap = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,1,20,37,37,37,37,32,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,36,158,215,217,217,217,
          202,95,22,5,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,33,79,175,246,254,254,254,254,253,232,173,127,113,33,1,0,0,0,0,0,0,
          0,0,0,0,0,1,35,163,218,251,254,253,234,222,234,253,254,252,249,243,163,33,0,0,0,0,0,0,0,0,0,0,0,9,91,219,245,
          254,254,244,177,143,177,244,254,254,254,253,218,79,3,0,0,0,0,0,0,0,0,0,7,90,219,253,254,253,221,127,34,11,34,
          126,208,247,254,254,251,170,21,0,0,0,0,0,0,0,0,1,47,207,253,254,254,232,95,10,0,0,0,8,48,177,252,254,254,215,
          37,0,0,0,0,0,0,0,3,22,159,247,254,254,252,173,22,0,0,0,0,0,4,117,246,254,254,217,37,0,0,0,0,0,0,0,34,95,232,
          254,254,254,251,142,10,0,0,0,0,0,3,95,238,253,254,203,32,0,0,0,0,0,0,3,79,159,247,254,255,254,252,173,22,0,0,0,
          0,0,4,115,246,254,252,172,21,0,0,0,0,0,1,36,175,232,254,254,255,254,254,232,82,2,0,0,0,0,4,127,250,254,250,129,
          5,0,0,0,0,1,35,163,246,253,254,253,254,255,254,249,124,4,0,0,0,0,4,127,250,254,250,127,4,0,0,0,0,20,158,245,254,
          253,223,205,252,255,254,233,82,2,0,0,0,0,5,129,250,254,250,127,4,0,0,0,0,37,215,254,247,221,96,136,250,254,254,
          217,39,0,0,0,0,0,21,172,252,254,249,125,4,0,0,0,0,37,215,254,222,141,16,127,250,254,254,203,32,0,0,0,0,0,37,215,
          254,254,232,82,2,0,0,0,0,32,203,253,217,129,10,127,250,254,252,172,21,0,0,0,0,0,37,217,254,254,208,46,0,0,0,0,0,
          9,140,250,217,127,9,127,250,254,250,129,5,0,0,0,0,0,37,213,251,247,138,9,0,0,0,0,0,4,127,250,222,141,16,129,250,
          254,250,127,4,0,0,0,0,0,18,110,158,156,65,2,0,0,0,0,0,4,127,250,247,221,112,179,252,254,245,114,4,0,0,0,0,0,0,4,
          16,16,2,0,0,0,0,0,0,4,113,242,254,253,239,248,254,250,222,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,77,206,247,253,
          253,254,254,250,217,39,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,47,163,218,251,254,252,241,201,32,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,1,33,79,170,213,172,115,77,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,21,36,21,4,2,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def writeBitmapToTempImage(bitmap):
    bitmap = [[pixel,pixel,pixel] for pixel in bitmap]
    bitmap = np.asarray(bitmap).reshape([28,28,3])    
    cv2.imwrite("one1.png", bitmap)

def readIntoRGBGrayscale():
    bitmap = cv2.imread('one1.png')
    return bitmap

def liniarize_image(bitmap):
    liniar = bitmap.reshape([100,-1])
    #liniar = liniar.tolist()
    return liniar


absolute_sepia = np.array([201,179,140])
def distanceToAbsoluteSepia_image(bitmap):
    img = []
    for line in bitmap:
        for pixel in line:
            r = pixel[0]
            g = pixel[0]
            b = pixel[0]
            the_color = np.asarray([r,g,b])
            chromatic_distance = 1 - sum(abs(real - sepia) for real,sepia in zip(absolute_sepia,the_color)) / 255
            img.append(chromatic_distance)
    img = np.asarray(img).reshape([100,1])
    return img

def sepia_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sepia_lower = np.array([np.round( 30 / 2), np.round(0.10 * 255), np.round(0.10 * 255)])
    sepia_upper = np.array([np.round( 45 / 2), np.round(0.60 * 255), np.round(0.90 * 255)])
    return cv2.inRange(hsv, sepia_lower, sepia_upper)

def readIntoSepia():
    sepia_kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
    bitmap = cv2.imread('one1.png')
    bitmap = np.flip(np.array(bitmap), 2)
    sepia = cv2.transform(bitmap, sepia_kernel)
    cv2.imshow("sepia", sepia)
    cv2.waitKey(0)
    return sepia


sepia_kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])

def toSepia(photo):
    sepia = cv2.transform(photo, sepia_kernel)
    #cv2.imshow("sepia", sepia)
    #cv2.waitKey(0)
    return sepia

def reshape_dataset_only_once_call_ever():
    f = open('pokemon.csv', 'r')
    line = f.readline() # ignore first line
    while True:
        line = f.readline()
        parts = line.strip().split(",")
        if len(parts) <= 1:
            break
        img = cv2.imread('images/'+parts[0]+".png")
        dim = (10, 10)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('resized/'+parts[0]+'.png', resized)
        cv2.imwrite('resized/'+parts[0]+'_sepia.png', toSepia(resized))
    f.close()
    pass


def loadPokemons():
    f = open('pokemon.csv', 'r')
    line = f.readline() # ignore first line
    
    training_inputs = []
    training_results = []
    
    while True:
        line = f.readline()
        parts = line.strip().split(",")
        if len(parts) <= 1: # or parts[0] == 'dartrix': # dartrix does not exist... 404
            break
        
        img = cv2.imread('resized/'+parts[0]+".png")
        if img is None:
            continue
        training_inputs.append(liniarize_image(img))
        training_results.append(0)
        
        img = cv2.imread('resized/'+parts[0]+"_sepia.png")
        training_inputs.append(liniarize_image(img))
        training_results.append(1)
        
    training_data = zip(training_inputs, training_results)
    f.close()
    
    training_data = list(training_data)
    spl = len(training_data)*80//100
    testing_data = training_data[spl:]
    training_data = training_data[:spl] 
    
    validation_data = []
    for tup in testing_data:
        vectorized_rez = tup[1]
        rez = (np.argmax(vectorized_rez))
        validation_data.append((tup[0],np.int64(rez)))

    
    return training_data, validation_data, testing_data










def loadImagesSimple():
    
    training_inputs = []
    training_results = []
    
    for index in range(220):
        img = cv2.imread('resized/images'+str(index)+".png")
        if img is None:
            continue
        
        img = distanceToAbsoluteSepia_image(img)
        img = liniarize_image(img)
        #img = np.asarray([pixel/255 for pixel in img])
        training_inputs.append(img)
        training_results.append(0)
        
        img = cv2.imread('resized/images'+str(index)+"_sepia.png")
        img = distanceToAbsoluteSepia_image(img)
        img = liniarize_image(img)
        #img = np.asarray([pixel/255 for pixel in img])
        training_inputs.append(img)
        training_results.append(1)
        
    training_data = zip(training_inputs, training_results)
    
    training_data = list(training_data)
    spl = len(training_data)*80//100
    testing_data = training_data[spl:]
    training_data = training_data[:spl] 
    
    validation_data = []
    for tup in testing_data:
        vectorized_rez = tup[1]
        rez = (np.argmax(vectorized_rez))
        validation_data.append((tup[0],np.int64(rez)))

    
    return training_data, validation_data, testing_data


















def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop



def gener8Dataset():
    counter = 220
    
    for imageCounter in range(1,12):
        example_image = cv2.imread('images (' + str(imageCounter) + ').jpg')
        #example_image = np.random.randint(0, 256, (1024, 1024, 3))
        
        for _ in range(10):
            random_crop = get_random_crop(example_image, 10, 10)
            cv2.imwrite("resized/images" + str(counter) + ".png", random_crop)
            cv2.imwrite("resized/images" + str(counter) + "_sepia.png", toSepia(random_crop))
            counter += 1
    pass


if __name__ == '__main__':
    '''writeBitmapToTempImage(our_secret_bitmap)
    img = readIntoRGBGrayscale();
    cv2.imshow("ta daa", img)
    cv2.waitKey(0)
    img = liniarize_image(img)
    print(img)'''
    
    gener8Dataset()
    
    
    ##NEVER AGAIN
    #reshape_dataset_only_once_call_ever()
    #print("DONE")
    #loadPokemons()
    #reshape_dataset_only_once_call_ever()
    

