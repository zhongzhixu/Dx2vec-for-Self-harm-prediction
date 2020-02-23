"""
Created on Tue Apr 17 17:40:04 2018
@author: zhongzhi
"""
#! /usr/bin/env python
# -*- coding:utf-8 -*-
#!/usr/bin/env python
#coding=utf-8

import pandas as pd
import networkx as nx
from copy import deepcopy
import numpy as np
#----------hyper parameters---------
EPISODE = 3
NET_EMBED_DIM = 64

#-----------------------read data--------------
-----------------------------------------------

#==cases===
count = len(open('cases.csv', 'r').readlines())
m,n = int(count/EPISODE),EPISODE+1 #add 1 space to reserve multihot encode
vectors_cases = [[0 for i in range(n)] for j in range(m)]
f=open('cases.csv', 'r')
while True:
    line=f.readline()
    if not line: break
    l = line.strip().split(',')
    vectors_cases[int(l[0])][int(l[1])]=[float(e) for e in l[2:]]
f.close()
#==multihot
f=open('multihot_cases.csv', 'r')
while True:
    line=f.readline()
    if not line: break
    l = line.strip().split(',')
    vectors_cases[int(l[0])][EPISODE+1-1]=[int(e) for e in l[1:]]
f.close()

#==controls===
count = len(open('controls.csv', 'r').readlines())
m,n = int(count/EPISODE),EPISODE+1
vectors_controls = [[0 for i in range(n)] for j in range(m)]
f=open('controls.csv', 'r')
while True:
    line=f.readline()
    if not line: break
    l = line.strip().split(',')
    vectors_controls[int(l[0])][int(l[1])]=[float(e) for e in l[2:]]
f.close()
#==multihot
f=open('multihot_controls.csv', 'r')
while True:
    line=f.readline()
    if not line: break
    l = line.strip().split(',')
    vectors_controls[int(l[0])][EPISODE+1-1]=[int(e) for e in l[1:]]
f.close()

#==dataframe
temp=[]
for i in range(len(vectors_cases)):
    temp.append( vectors_cases[i][0] + vectors_cases[i][1] + vectors_cases[i][2] + vectors_cases[i][3])
dfcases = pd.DataFrame(
        data=temp,
        index = range(len(temp)),
        columns = range(len(temp[0]))
        )
age_cases=pd.read_csv('age_cases', header=None)
sex_cases=pd.read_csv('sex_cases', header=None)
dfcases['age']=age_cases[0]
dfcases['sex']=sex_cases[0]
dfcases['y'] = 1

temp=[]
for i in range(len(vectors_controls)):
    temp.append (vectors_controls[i][0] + vectors_controls[i][1] + vectors_controls[i][2]+ vectors_controls[i][3])
dfcontrols = pd.DataFrame(
        data=temp,
        index = range(len(temp)),
        columns = range(len(temp[0]))
        )
age_controls=pd.read_csv('age_controls', header=None)
sex_controls=pd.read_csv('sex_controls', header=None)
dfcontrols['age']=age_controls[0]
dfcontrols['sex']=sex_controls[0]

dfcontrols['y'] = 0
#%%
#-------------data formation -------------
------------------------------------------
random_s = 38
random_s=random_s+1
df = dfcases.append(dfcontrols)   
df = df.sample(frac=1, random_state=random_s).reset_index(drop=True) 
values = df.values
n_train = int(df.shape[0]*0.8)
train = values[:n_train, :]
test = values[n_train:, :]
# split into input and outputs
train_X, train_y = train[:, :64*3], train[:, -1]
test_X, test_y = test[:, :64*3], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 64, int(train_X.shape[1]/64)))
test_X = test_X.reshape((test_X.shape[0], 64, int(test_X.shape[1]/64) ))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#%%
#---------------IMPORT TOOLS--------
import keras
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Model

#%%
#====================DX2vec DNN=============
============================================
# build framework
main_input = Input(shape=(train_X.shape[1], train_X.shape[2]), name='m_input') #caution: train_X has three dimensions
lstm_out = GRU(3, name='lstm_out')(main_input)
dout = Dropout(0.01)(lstm_out)
auxiliary_input = Input(shape=(len(temp[0])-NET_EMBED_DIM*EPISODE+2,), name='auxiliary_input')
x = keras.layers.concatenate([dout, auxiliary_input])      
x = Dense(8)(x)

outputlayer = Dense(1, activation='sigmoid', name='output')(x)
# compile and fit
model = Model(inputs=[main_input, auxiliary_input], 
              outputs=outputlayer)
model.compile(optimizer='adam', loss='binary_crossentropy')
class_weight = {0: 0.1,
                1: 0.9}
history = model.fit([train_X, train_auxi_X], train_y, validation_data=([test_X, test_auxi_X], test_y),
                    epochs=5, batch_size=75, class_weight=class_weight)  
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# predict
yhat = model.predict([test_X, test_auxi_X]) 
yhat = [round(i[0]) for i in yhat]

print( classification_report(test_y, yhat,digits=3) )

# Output the Dx2vec embedding
from keras.models import Model
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('lstm_out').output)
intermediate_output = intermediate_layer_model.predict([train_X, train_auxi_X])
THREED_PATH = 'C:/Users/Documents/three_dim_visualization/'
np.savetxt(THREED_PATH+"dx2vec_emb.csv", intermediate_output, delimiter=",")
np.savetxt(THREED_PATH+"dx2vec_emb_class.csv", train_y)

#%%
#-----------------BASELINE DNN---------------
---------------------------------------------
auxiliary_input = Input(shape=(len(temp[0])-NET_EMBED_DIM*EPISODE+2,), name='auxiliary_input')

# design network
class_weight = {0: 0.18,
                1: 0.82}
model = Sequential()

model.add(Dense(16, input_shape=(len(temp[0])-NET_EMBED_DIM*EPISODE+2,)))
model.add(Dense(8))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
# fit network
history = model.fit(train_auxi_X, train_y, epochs=5, batch_size=75, validation_data=(test_auxi_X, test_y), 
                    verbose=2, shuffle=False, class_weight=class_weight)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# PREDICT 
yhat = model.predict(test_auxi_X)
yhat = [round(i[0]) for i in yhat]

print( classification_report(test_y, yhat,digits=3) )

#%%
#------------p-r/roc curve---------------
-----------------------------------------

from sklearn.metrics import precision_recall_curve, roc_curve, classification_report, roc_auc_score      
def plot_pr(auc_score, precision, recall, label=None, pr=True):  
    pylab.figure(num=None, figsize=(6, 5))  
    pylab.xlim([0.0, 1.0])  
    pylab.ylim([0.0, 1.0])  
     
    if pr:
        pylab.xlabel('Recall') 
        pylab.ylabel('Precision')
        pylab.title('P/R CURVE') 
    else:
        pylab.xlabel('FPR') 
        pylab.ylabel('TPR')
        pylab.title('ROC CURVE') 
     
    pylab.fill_between(recall, precision, alpha=0.5)  
    pylab.grid(True, linestyle='-', color='0.75')  
    pylab.plot(recall, precision, lw=1)      
    pylab.show()
precision, recall, thresholds = precision_recall_curve(test_y, y_pred_prob)
fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob, pos_label=1)
plot_pr(0.5,  precision,recall, "pos")
plot_pr(0.5,  tpr,fpr, "pos",pr=False)




