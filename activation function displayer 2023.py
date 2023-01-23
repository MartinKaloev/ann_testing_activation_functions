"""
I.Contacts:
Contact me at kaloev_92@mail.ru
eng. M. Kaloev

II.how to start
1.Check important versions (this is automaticaly done for cloud version)
2.Press Run Button
3.Collect the activation function signals from the dir: results

III.What code does
1.Creates model of neural network
2.Generates traing data (random data)
3.Model is trained and it weights are saved
4.Signals output of hiden layers are displayed 
4.1 signal of Relu is used for traing
4.2 signal of all other activation function is shown after relu is changed for other functions in longest layer (not trained)


IV.important versions:
keras: 2.6.0
numpy: 1.19.5
tensorflow: 2.6.0
python 3.9

V.model:
Model: "sequential"
____________________________
input shape: (40)
_____________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 60)                2460      
_________________________________________________________________
dense_1 (Dense)              (None, 80)                4880      
_________________________________________________________________
dense_2 (Dense)              (None, 100)               8100      
_________________________________________________________________
dense_3 (Dense)              (None, 75)                7575      
_________________________________________________________________
dense_4 (Dense)              (None, 50)                3800      
_________________________________________________________________
dense_5 (Dense)              (None, 25)                1275      
_________________________________________________________________
dense_6 (Dense)              (None, 5)                 130       
=================================================================
Total params: 28,220
Trainable params: 28,220
Non-trainable params: 0

VI.discussion

Q1. there are huge areas with out put of zeros:
A: yes, this expected and downside of using relu activation

Q2. why leakyRelu is NOT having same output as relu if they are same "famillies":
A: leakyrelu and relu behave in same way only for positives outputs

Q3. sometimes sigmoid can swaped  relu (sigmoid -> relu) in hiden layer with same weight  withot changing results, 
and some time is (tanh-relu), why?
A: Do check the training information for the neutwork, if "loss" is close to 0 and "acc" is 1.000 from the epoch 20+, this may be because the
NN is feed with very similar information and is overspecilised (overftted), in this case the "noise" signal that tanh creates with similar to 
bad signal relu produces in the hiden layers. This effect is discussed in the papper. 

Q4. some functions are causing masive "explosions" of signals bigger than relu?
A: yes, this is to expected, please do read documentation for: exponential activation function

Q5. my system is laging on some actiavtion functions and produces realy bad signal with valuse [-0.3 to 0.3]
A: yes, this is to expected, some activation function are use primary for output layers in NN for clasification
do read documentation for : softmax, softsign

Q6. some signals have very low, negative values?
A: Linear activation function does NOT change input-outputs.

VII.disclaimer
This code and the paper it supports are focused on signals that activation functions provides in hiden layers
Based on the code NO generalised conclusions should be made for the topics as: size of neural networks, overfititing problems, selecting good traing data, learning rate, epochs,
or other parameters for building of NN. 


ps: i am dislexyc , if  mispeleed something.

"""
import matplotlib.pyplot as plt
import numpy
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
from keras.models import Sequential
from numpy import loadtxt
from numpy import loadtxt
from keras.layers import Dense
import numpy as np
import keras as K
import math
import random
import os

print(keras.__version__)
print(numpy.__version__)
print(tensorflow.__version__)


def setup_dir():
    isExist = os.path.exists('results')
    if isExist !=True:
        os.mkdir('results')
    isExist = os.path.exists('results/layer 1')
    if isExist !=True:
        os.mkdir('results/layer 1')
    
    isExist = os.path.exists('results/layer 2')
    if isExist !=True:
        os.mkdir('results/layer 2')
    isExist = os.path.exists('results/layer 3')
    if isExist !=True:
        os.mkdir('results/layer 3')


def plts(layer_,name_fig):
    plt.xlabel("Neuron number ")
    plt.ylabel("Value of output of neuron")
    plt.title(name_fig)
    plt.plot(layer_)
    plt.savefig(name_fig) 
    plt.close()



class act_v_caller:

    def __init__(self):
        self._weights_saver =None
        self.trainset_input = []
        self.trainset_output =[]
        self.evalulation_set =[]
        self.out_put_softmaxs =[]



    def gens(self, cut_=40, begin_=1, end_=10):
        randomlist = []
        for i in range(0,cut_):
            n = random.randint(begin_,end_)
            randomlist.append(n)
        return randomlist

    def generate_train_set(self):
        #ge sets
        
        for i in range(0, 55):
            self.trainset_input.append(self.gens())
        
        
        
        z = self.gens(55,0,4)
        for i in z:
            x = [0]*5
            x[i]=1
            self.trainset_output.append(x)
    
  
        
    def generate_eval_set(self):
        #ge sets
       
        for i in range(0, 10):
            self.evalulation_set.append(self.gens())


    def gen_model_with_weights(self):
        #gen mocel
        model = Sequential()
        model.add(	Dense(60, input_dim=40, activation='relu'))
        model.add(	Dense(80, activation='relu'))
        model.add(	Dense(100, activation='relu'))	
        model.add(	Dense(75, activation='relu'))	
        model.add(	Dense(50, activation='relu'))
        model.add(  Dense(25, activation='relu'))
        model.add(	Dense(5, activation='softmax'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(self.trainset_input, self.trainset_output, epochs=100, verbose=1)
        self._weights_saver=model.get_weights()

    def geen_test(self, list_func=['relu','tanh','sigmoid','LeakyReLU','elu','selu','exponential','softsign', 'softplus', 'softmax', 'linear']):
        print(list_func)
        for func_ in list_func:
            model2 = Sequential()
            
            model2.add(	Dense(60, input_dim=40, activation='relu'))
            model2.add(	Dense(80, activation='relu'))
            model2.add(	Dense(100, activation=func_))	
            model2.add(	Dense(75, activation='relu'))	
            model2.add(	Dense(50, activation='relu'))
            model2.add(  Dense(25, activation='relu'))
            model2.add(	Dense(5, activation='softmax'))
            #model2.summary()

            model2.set_weights(self._weights_saver)
            model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            #puts extra models here, bcz i am so lazy to find difrent way for implementations, lol shagy 

            model3 = Sequential()
            model3.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model3.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))


            model4 = Sequential()
            model4.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model4.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))
            model4.add(	Dense(100, weights=model2.layers[2].get_weights(), activation=func_))	
     

            model5 = Sequential()
            model5.add(	Dense(60, weights=model2.layers[0].get_weights(), input_dim=40, activation='relu'))
            model5.add(	Dense(80, weights=model2.layers[1].get_weights(), activation='relu'))
            model5.add(	Dense(100, weights=model2.layers[2].get_weights(), activation=func_))	
            model5.add(	Dense(75, weights=model2.layers[3].get_weights(), activation='relu'))	

            for example in range( len(self.evalulation_set) ):
                cl=model2.predict([self.evalulation_set[example]])
                cl=cl[0]
                cl=np.round(cl)
                print("function is: ",func_, "example numb: ", example, "prediction is: ", cl )
                if example==4 and func_=='relu':
                   
                    out1=model3.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 80: ", out1[0].tolist() )
                    name2="results/layer 1/output values for act fun "+str(func_)+" for layer with lenght 80 "
                    plts(out1[0].tolist(),name2)
                    print(name)
                if example ==4: 
                    
                    out2=model4.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 100: ", out2[0].tolist() )
                    name2="results/layer 2/output values for act fun "+str(func_)+ " for layer with lenght 100 "
                    print(name)
                    plts(out2[0].tolist(),name2)

                    out3=model5.predict([self.evalulation_set[example]])
                    name=("output values for act fun: ",func_, "for layer with lenght 75: ", out3[0].tolist() )
                    name2="results/layer 3/output values for act fun "+str(func_)+ " for layer with lenght 75 "
                    print(name)
                    plts(out3[0].tolist(),name2)
                    

setup_dir()
test= act_v_caller()
test.generate_train_set()
test.generate_eval_set()
test.gen_model_with_weights()
test.geen_test()


