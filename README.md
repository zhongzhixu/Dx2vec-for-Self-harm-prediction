# Dx2vec
This repository is the implementation of Dx2vec in paper "Predicting Self-harm Incidents Following Inpatient Visits using Disease Comorbidity Network".

Dx2vec is a novel deep learning model which embeds an individual's structural information into a low dimensional vector. This vector can then be used in various downstream tasks such as disease risk modelling, patient clustring and the estimation of hospital utilization. 
In this paper we use Dx2vec to perform the self-harm prediction after hospital discharge.

![figure](https://github.com/zhongzhixu/Dx2vec-for-Self-harm-prediction/blob/master/architecture_multi_input.jpg)

The data is provided by the Hospital Authority of Hong Kong the ethical approval UW11-495. The data can not be made available to others according to the Hospital Authority and the ethical approval. Instead, we provide some simulated cases in DATA folder.  

## Files in the folder
DATA: simulated cases

DX2vec.py

## Environment:
Python 3.6

Keras 2.2.4

TensorFlow 1.13.1

## Running the code

Clone all files to the local computer and run







