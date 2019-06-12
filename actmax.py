from vis.utils import utils
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import activations
import os
from sklearn.metrics import accuracy_score
from vis.visualization import visualize_cam
import keras.backend as K
import sys
import json
from keras.applications.vgg16 import preprocess_input
from vis.visualization import visualize_activation
from matplotlib import pyplot as plt
from PIL import Image

def visualize(layer_name, filter_index):

    layer_output = layer_dict[layer_name].output
    #defining a loss function that maximizes mean of output of filter
    loss = K.mean(layer_output[:, filter_index, :, :])

    #random input image
    input_img = np.random.random((1, 640, 360))
    input_img *= 255

    # we compute the gradient of the input picture wrt this loss and normalize
    grads = K.gradients(loss, input_img)[0]
    grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    iterate = K.function([input_img], [loss, grads])

    iterations = 200
    ascent_step = 0.01
    for i in range(iterations):
        loss_value, grads_value = iterate([input_img_data])
        input_img += grads_value * step
    
    return input_img


model = keras.models.load_model('models/arrow.h5')

layer_idx = utils.find_layer_idx(model, 'dense_2')
model.layers[19].activation = activations.linear
model = utils.apply_modifications(model)

final = np.array([])

for i in range(0,3):
    img = visualize_activation(model, 19, filter_indices=i)
    if(i==0):
        final1=img
    final1 = np.concatenate((final1, img), axis=0)

for i in range(4,7):
    img = visualize_activation(model, 19, filter_indices=i)
    if(i==4):
        final2=img
    final2 = np.concatenate((final2, img), axis=0)


for i in range(8,11):
    img = visualize_activation(model, 19, filter_indices=i)
    if(i==8):
        final3=img
    final3 = np.concatenate((final3, img), axis=0)


for i in range(12,15):
    img = visualize_activation(model, 19, filter_indices=i)
    if(i==12):
        final4=img
    final4 = np.concatenate((final4, img), axis=0)

final_frame = np.concatenate((final1, final2, final3, final4), axis=1)


cv2.imwrite('veamos.png', final_frame)




