# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:33:40 2023

@author: wmr_w
"""

import cv2 
import numpy as np


#img = cv2.imread('C:/Users/wmr_w/weimar/MAESTRIA/MCC_VISION/Proyecto Final/data/20230505_101221.jpg')


def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    treshold,_ = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    
    mask = np.uint8(1*(gray<treshold))
    
    #caracteristicas
    
    B=(1/255)*np.sum(img[:,:,0]*mask)/np.sum(mask)
    G=(1/255)*np.sum(img[:,:,1]*mask)/np.sum(mask)
    R=(1/255)*np.sum(img[:,:,2]*mask)/np.sum(mask)
    return [B,G,R]

#gerenarion del Dataset de caracteristicas
import glob

paths=[
       'C:/Users/wmr_w/weimar/MAESTRIA/MCC_VISION/Proyecto Final/data/Conforme2/' #-1
       ,'C:/Users/wmr_w/weimar/MAESTRIA/MCC_VISION/Proyecto Final/data/Noconforme2/'
       ]

labels =[]
features=[]
    
for label, path in enumerate(paths):
    for filename in glob.glob(path+"*.JPG"):
        #print(filename)
        img=cv2.imread(filename)
        features.append(getFeatures(img))
        labels.append(label)

features =np.array(features)
labels=np.array(labels)
labels=2*labels-1

#"visualizacion de caracteristicas"

import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

for i, features_row in enumerate(features):
    if labels[i]==-1:
        ax.scatter(features_row[0],features_row[1],features_row[2],marker='*',c='k')
    else:
        ax.scatter(features_row[0],features_row[1],features_row[2],marker='*',c='r')

ax.set_xlabel('B')
ax.set_xlabel('G')
ax.set_xlabel('R')


# error en funciion de las constantes del hiperplano 
#dos caracteristicas

subFeatures = features[:,1::]
loss=[]

for w1 in np.linspace(-6,6,100):
    for w2 in np.linspace(-6,6,100):
        totalError=0
        for i,feature_row in enumerate(subFeatures):
            sample_error=(w1*feature_row[0]+w2*feature_row[1]-labels[i])**2
            totalError += sample_error
        loss.append([w1,w2,totalError])

loss = np.array(loss)

from matplotlib import cm
fig = plt.figure()
ax1=fig.add_subplot(111,projection='3d')

ax1.plot_trisurf(loss[:,0],loss[:,1],loss[:,2],cmap=cm.jet, linewidth=0)

ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('loss')


# Calculo del hiperplano que separe las dos calses de forma optima

A=np.zeros((4,4))
b=np.zeros((4,1))

for i, feature_row in enumerate(features):
    x=np.append([1],feature_row)
    x=x.reshape((4,1))
    y=labels[i]
    A=A+x*x.T
    b=b+x*y
    
invA=np.linalg.inv(A)

W=np.dot(invA, b)
 
X=np.arange(0,1,0.1)
Y=np.arange(0,1,0.1)
X,Y=np.meshgrid(X,Y)

#ax+by=0
#W[3]*Z+W[1]*X+W[0]=0
Z=(W[1]*X+W[2]*Y+W[0])/W[3]

ax.plot_surface(X,Y,Z, cmap=cm.Blues)

#error del entrenamiento 

prediction = 1*(W[0]+np.dot(features,W[1::]))>=0

prediction = 2*prediction-1

error = np.sum(prediction != labels.reshape(-1,1))/len(labels)

efectividad = 1- error


#prediccion para una sola imagen

path_img = 'C:/Users/wmr_w/weimar/MAESTRIA/MCC_VISION/Proyecto Final/data/Noconforme/20230505_100728.jpg'

img=cv2.imread(path_img)

feature_vector=np.array(getFeatures(img))

result=np.sign(W[0]+np.dot(feature_vector,W[1::]))

if result == -1:
    print('Conforme')
else:
    print('No Conforme')





















