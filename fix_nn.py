#Model.layers[index].output

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from numpy import loadtxt
from numpy import loadtxt
from keras.layers import Dense
import numpy as np
import keras as K
import math
import random

def plts(layer_,name_fig):
    plt.xlabel("Value")
    plt.ylabel("neuron number")
    plt.title(name_fig)
    plt.plot(layer_)
    plt.savefig(name_fig) 
      




class act_v_caller:

    def __init__(self):
        self._weights_saver =None
        self.trainset_input = []
        self.trainset_output =[]
        self.evalulation_set =[]
        self.out_put_softmaxs =[]

    def gen_fig(self):
        #generate figs<<<
        print("do")

    def gen_fig2(self):
        #generate figs<<<
        print("do")

    def gens(self, cut_=40, begin_=1, end_=10):
        randomlist = []
        for i in range(0,cut_):
            n = random.randint(begin_,end_)
            randomlist.append(n)
        return randomlist

    def generate_train_set(self):
        #ge sets
        print("do")
        for i in range(0, 55):
            self.trainset_input.append(self.gens())
        
        
        
        z = self.gens(55,0,4)
        for i in z:
            x = [0]*5
            x[i]=1
            self.trainset_output.append(x)
    
  
        
    def generate_eval_set(self):
        #ge sets
        print("do")
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

            for example in range( len(self.evalulation_set) ):
                cl=model2.predict([self.evalulation_set[example]])
                cl=cl[0]
                cl=np.round(cl)
                print("function is: ",func_, "example numb: ", example, "prediction is: ", cl )
                if example==4:
                    pre_long=model2.layers[1].output
                    print(pre_long)
                  
                    name=("output values for act fun: ",func_, "for layer with lenght 80" )
                    plts(pre_long, name )


test= act_v_caller()
test.generate_train_set()
test.generate_eval_set()
test.gen_model_with_weights()
test.geen_test()


