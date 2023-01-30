import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

""""
# to add
#show pytorch version
#check GPU up
#check how to firwads
#how to randoms
#how evals
#how to mods
#add sys
#plts

"""
print(torch.__version__)

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
train_outputs = []
z = gens(10,0,4)
for i in z:
    x = [0]*5
    x[i]=1
    #x=torch.tensor(x)
    train_outputs.append(x)
train_outputs = torch.tensor(train_outputs , dtype=torch.float32)
print("trains: ", train_outputs)
train_inputs = torch.tensor(np.random.rand(10, 40), dtype=torch.float32)
train_eval = torch.tensor(np.random.rand(1, 40), dtype=torch.float32)


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
        self.activation_function = activation_function

    def forward(self, x):
        x = F.relu(self.fc1(x))
        lock1 = F.relu(self.fc2(x))
        lock2 = self.activation_function(self.fc3(lock1))
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

# Initialize model1 with ReLU activation
model1 = NeuralNet(F.relu)
# Initialize model2 with Sigmoid activation
model2 = NeuralNet(F.sigmoid)
# Initialize model2 with Tanh activation
model3 = NeuralNet(F.tanh)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.001)



# Train model1
for epoch in range(25):
    optimizer.zero_grad()
    _,_,_, output = model1(train_inputs)
    #print("trues: ",output)
    loss = criterion(output, train_outputs) #output, train_ouputs
    loss.backward()
    optimizer.step()

# Transfer weights from model1 to model2
model2.transfer_weights(model1)
model3.transfer_weights(model1)

models=[]
models.append(model1)
models.append(model2)
models.append(model3)


#prints
for example in train_eval:
    for mod in models:
        mod.eval()
        lay1, lay2, lay3, out = mod(example)
        print("funs: ", str(mod.activation_function))
        print(lay1.tolist())
        print(lay2.tolist())
        print(lay3.tolist())
        print(out.tolist())
        #plts2(lay1)
        #plts2(lay2)
        #plts2(lay3)
