import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import keras.backend as K
import sys
import json
from keras.applications.vgg16 import preprocess_input
from PIL import Image

def rgb2gray(rgb):
    x = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    x = x/255
    return np.expand_dims(x, axis=2)

N = 30 #number of figures to generate

print('Loading model...')
model = keras.models.load_model('models/arrow.h5')
print(model.summary())

print('Reading files...')
with open('data/X_train_arrow.txt', 'r') as filehandle:
    X_train = json.load(filehandle)
with open('data/y_train_arrow.txt', 'r') as filehandle:
    y_train = json.load(filehandle)
X_train = np.array(X_train)
y_train = np.array(y_train)

#choose target layer to visualize activations from
target_layer = model.get_layer('<target-layer-name>')

imgs = np.array([rgb2gray(np.array(Image.open('<path-to-rico-dataset>/combined/{}.jpg'.format(photo_path)).resize((360, 640)))) for photo_path in X_train[:N]])
y_train = y_train[:N]

for imindex in range(N):

    cvimg = cv2.imread('<path-to-rico-dataset>/combined/{}.jpg'.format(str(X_train[imindex])))
    x = np.expand_dims(imgs[imindex], axis=0)

    #forward propagate input
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]

    #compute gradients of output wrt target layer
    grads = K.gradients(class_output, target_layer.output)[0]

    #GAP gradients
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, target_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    #weight feature maps by importance
    for i in range(64):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    #average weighted feature maps
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    #ReLU heatmap
    heatmap = np.maximum(heatmap, 0)
    out=heatmap
    np.divide(heatmap, np.max(heatmap), out=out, where=np.max(heatmap)!=0)
    heatmap=out

    #upscale heatmap into input size and range
    heatmap = cv2.resize(heatmap, (cvimg.shape[1], cvimg.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    #apply cool colormap and superimpose it to input img
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    #otsu threshold heatmap into binary
    heatgray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(heatgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #find largest blob and bounding rectangle
    im, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    x, y, w, h = cv2.boundingRect(maxContourData)

    cv2.rectangle(finalImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite('figs/{}heat.png'.format(imindex), heatmap)

    cv2.imwrite('figs/{}thres.png'.format(imindex), th2)

    #add bounding box to threshold image
    th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(th2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('figs/{}thresbb.png'.format(imindex), th2)

    #add bounding box to input image
    cv2.rectangle(cvimg, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('figs/{}final.png'.format(imindex), cvimg)
