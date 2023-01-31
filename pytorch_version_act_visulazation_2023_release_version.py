import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os

""""
how it works:
1.creates model of ANN for classificaton (see Cell with header #Model preparation)
2.Data for training and data for testing is generated
3.ANN is tested
4.Visualisation of outpust of 3 hiddens layers are tested
5.1.Test are provided to check what will happen if same model uses diffrent activation func without re-traing.
5.2.How this will effect outputs of hidden layers and classiicataion (Cell with header #Setup testing act func )
6.Fig with outputs of layers in dir results (#prints outputs, save them in results dir)


"""

#print versions 
print("pytorch version: ", torch.__version__)



#setup directory
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
    isExist = os.path.exists('results/out layer')
    if isExist !=True:
        os.mkdir('results/out layer')

#def function for visualization 
def plts(layer_,name_fig):
    plt.xlabel("Neuron number ")
    plt.ylabel("Value of output of neuron")
    plt.title(name_fig)
    plt.plot(layer_)
    plt.savefig(name_fig) 
    plt.close()




#Check Cuda paths #Check GPU #Check NVIDIA drivers 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("PyTorch is using GPU")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU")

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")


#Generate randomse
def gens(cut_=40, begin_=1, end_=10):
    randomlist = []
    for i in range(0,cut_):
        n = random.randint(begin_,end_)
        randomlist.append(n)
    return randomlist

#Generate train data
#Gen class labels
train_outputs = []
z = gens(10,0,4)
for i in z:
    x = [0]*5
    x[i]=1
    train_outputs.append(x)
train_outputs = torch.tensor(train_outputs , dtype=torch.float32)
print("trains: ", train_outputs)

#Gen tensor input
train_inputs=[]
for i in range(0, 10):
    train_inputs.append(gens(40,0,100))
train_inputs = torch.tensor(train_inputs,dtype=torch.float32)
print("traing inputs: ", train_inputs)

#Gen evalation tests
train_eval=[]
for i in range(0,2):
    train_eval.append(gens(40,0,100))
train_eval = torch.tensor(train_eval,dtype=torch.float32)


#Model preparation
class NeuralNet(nn.Module):
    def __init__(self, activation_function):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(40, 60) #relu
        self.fc2 = nn.Linear(60, 80) #relu
        self.fc3 = nn.Linear(80, 100) #custom
        self.fc4 = nn.Linear(100, 75) #relu
        self.fc5 = nn.Linear(75, 50)  #relu
        self.fc6 = nn.Linear(50, 25)  #relu
        self.fc7 = nn.Linear(25, 5)  #relu
        self.activation_function_ = activation_function
        self.name_act_f = activation_function.__name__

    def forward(self, x):
        x = F.relu(self.fc1(x))
        lock1 = F.relu(self.fc2(x))
        lock2 = self.activation_function_(self.fc3(lock1))
        lock3 = F.relu(self.fc4(lock2))
        x = F.relu(self.fc5(lock3))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return lock1, lock2, lock3, x

    def transfer_weights(self, model):
        self.fc1.weight = model.fc1.weight
        self.fc1.bias = model.fc1.bias
        self.fc2.weight = model.fc2.weight
        self.fc2.bias = model.fc2.bias
        self.fc3.weight = model.fc3.weight
        self.fc3.bias = model.fc3.bias
        self.fc4.weight = model.fc4.weight
        self.fc4.bias = model.fc4.bias
        self.fc5.weight = model.fc5.weight
        self.fc5.bias = model.fc5.bias
        self.fc6.weight = model.fc6.weight
        self.fc6.bias = model.fc6.bias
        self.fc7.weight = model.fc7.weight
        self.fc7.bias = model.fc7.bias

#Setup testing act func
# Initialize model1 with ReLU activation
model1 = NeuralNet(F.relu)
# Initialize model2 with Sigmoid activation
model2 = NeuralNet(F.sigmoid)
# Initialize model3 with Tanh activation
model3 = NeuralNet(F.tanh)
# Initialize model4 with leakyRelU activation
model4 = NeuralNet(F.leaky_relu)
# Initialize model4 with exponational activation
model5 = NeuralNet(torch.exp)
# Initialize model6 with softamx activation
model6 = NeuralNet(F.softmax)
# Initialize model7 with linear activation
model7 = NeuralNet(F.elu)



# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.0001)



# Train model1
for epoch in range(100):
    optimizer.zero_grad()
    _,_,_, output = model1(train_inputs)
    
    loss = criterion(output, train_outputs) #output, train_ouputs
    loss.backward()
    optimizer.step()

# Transfer weights from model1 to model2
model2.transfer_weights(model1)
model3.transfer_weights(model1)
model4.transfer_weights(model1)
model5.transfer_weights(model1)
model6.transfer_weights(model1)
model7.transfer_weights(model1)

models=[]
models.append(model1)
models.append(model2)
models.append(model3)
models.append(model4)
models.append(model5)
models.append(model6)
models.append(model7)


#prints outputs, save them in results dir
setup_dir()
for example in train_eval:
    model1.eval()
    lay1, _, _, _=  model1(example)
    name2_80="results/layer 1/output values for act fun "+str(model1.name_act_f)+" for layer with lenght 80 "
    plts(lay1.tolist(), name2_80)
    for mod in models:
        mod.eval()
        lay1, lay2, lay3, out = mod(example)
        print("funs: ", str(mod.name_act_f))
        name2_80="results/layer 1/output values for act fun "+str(mod.name_act_f)+" for layer with lenght 80 "
        name2_100="results/layer 2/output values for act fun "+str(mod.name_act_f)+" for layer with lenght 100 "
        name2_75="results/layer 3/output values for act fun "+str(mod.name_act_f)+" for layer with lenght 75 "
        name2_out="results/out layer/output values for act fun "+str(mod.name_act_f)+" for layer classification "
        print(lay1.tolist())
        print(lay2.tolist())
        print(lay3.tolist())
        print(out.tolist())
        
        plts(lay2.tolist(), name2_100)
        plts(lay3.tolist(), name2_75)
        plts(out.tolist(), name2_out )
       