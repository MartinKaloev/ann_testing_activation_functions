from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from numpy import loadtxt
from numpy import loadtxt
from keras.layers import Dense

import keras as K

########
###works with tf=2.3.1 and neet dataset and dataet1 filles with 41 entries?, yeah
#########



def run(a,a2):
	model = Sequential()
	model.add(	Dense(60, input_dim=40, activation='relu'))
	model.add(	Dense(80, activation='relu'))
	model.add(	Dense(100, activation='relu'))	
	model.add(	Dense(75, activation='relu'))	
	model.add(	Dense(50, activation='relu'))
	model.add(  Dense(25, activation='relu'))
	model.add(	Dense(5, activation='softmax'))

	dataset = loadtxt('dataset', delimiter=',')
	dataset2 = loadtxt('dataset2', delimiter=',')
	z = dataset[:,0:40]

	Y = dataset[:,40]
	X = dataset[:,0:40]
	X2 = dataset2[:,0:40]
	print(Y)
	print(z)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(z, Y, epochs=100, verbose=0)
	model.fit(z, Y, epochs=150, batch_size=10, verbose=0, validation_data=(X,Y))
	model.summary()
	predict_ = model.predict(X2)
	predictions = model.predict_classes(X2)




	#hiden_l_100;

	model2 = Sequential()
	model2.add(	Dense(60, weights=model.layers[0].get_weights() , input_dim=40, activation='relu'))
	model2.add(	Dense(80, weights=model.layers[1].get_weights() , activation='relu'))
	model2.add(	Dense(100, weights=model.layers[2].get_weights() , activation=a))	


	model2.summary()
	#hiden_layer_75
	model3 = Sequential()
	model3.add(	Dense(60, weights=model.layers[0].get_weights() , input_dim=40, activation='relu'))
	model3.add(	Dense(80, weights=model.layers[1].get_weights() , activation='relu'))
	model3.add(	Dense(100, weights=model.layers[2].get_weights() , activation=a))	
	model3.add(	Dense(75, weights=model.layers[3].get_weights() , activation='relu'))
	
	model3.summary()
	#hiden_layer_check

	model4 = Sequential()
	model4.add(	Dense(60, weights=model.layers[0].get_weights() ,input_dim=40, activation='relu'))
	model4.add(	Dense(80, weights=model.layers[1].get_weights() ,activation='relu'))
	model4.add(	Dense(100, weights=model.layers[2].get_weights() , activation=a))	
	model4.add(	Dense(75, weights=model.layers[3].get_weights() ,activation='relu'))	
	model4.add(	Dense(50, weights=model.layers[4].get_weights() ,activation='relu'))
	model4.add(  Dense(25, weights=model.layers[5].get_weights() ,activation='relu'))
	model4.add(	Dense(5, weights=model.layers[6].get_weights() ,activation='softmax'))



	model5 = Sequential()
	model5.add(	Dense(60, weights=model.layers[0].get_weights() , input_dim=40, activation='relu'))
	model5.add(	Dense(80, weights=model.layers[1].get_weights() , activation='relu'))
	model5.add(	Dense(100, weights=model.layers[2].get_weights() , activation='relu'))


	def write_down(name_file,a,a2,predict_):
			f=open(str(name_file),"a")
			f.write('layer for '+str(a)+str(a2)+'\n')
			print("\n")
			f.write(str(predict_))
			f.close()


	#orgininal nn;
	write_down("hard_og_r.txt",a,a2,model.predict_classes(X2))
	#checking last L_test_nn
	write_down("check_output_layer.txt",a,a2,model4.predict_classes(X2))
	#checking 100 leyar_
	write_down("main_h_l_check_100.txt",a,a2,model2.predict(X2))
	#checking 75_sub layer_
	write_down("sub_layer_75.txt",a,a2,model3.predict(X2))
	#comperance 100_leyear
	write_down("og_100_l.txt",a,a2,model5.predict(X2))

	model.summary()
	model2.summary()
	model3.summary()
	model4.summary()




def do(a):
	list_a=['relu', 'sigmoid', 'tanh']

	for A in list_a:

		run(A,a)

for i in range(20):
	do(i)
	print(i)