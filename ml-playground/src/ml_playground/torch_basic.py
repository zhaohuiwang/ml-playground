""""
https://github.com/PacktPublishing/
https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E

PyTorch is based on Torch, a framework for doing fast computation that is written in C. Torch has a Lua (means "Moon" in Portuguese) wrapper for constructing models.

One of the main special features of PyTorch is that it adds a C++ module for autodifferentiation to the Torch backend using torch.autograd engine. 
By default, PyTorch uses eager mode computation. Same as the Keras


PyTorch Ecosystem
fast.ai: An API that makes it straightforward to build models quickly.
TorchServe: An open-source model server developed in collaboration between AWS and Facebook.
TorchElastic: A framework for training deep neural networks at scale using Kubernetes.
PyTorch Hub: An active community for sharing and extending cutting-edge models.
TorchVison: A library dedicated to computer vision tasks that offers datasets, model architectures, and common image transformations.
TorchAudio: A library for audio processing and audio manipulation utilities.
TorchText: A natural language processing library that provides data processing utilities and popular datasets for the NLP field.

"""

import numpy as np
import torch

def tanh(x):
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
 return np.where(x>0,x,0)

def linear(x):
 return x

def softmax(x):
 return np.exp(x)/np.sum(np.exp(x))

def mse(p, y):
 return np.mean(np.square(p - y))

def mae(p, y):
 return np.mean(np.abs(p-y))

def binary_cross_entropy(p, y):
 return -np.mean((y*np.log(p)+(1-y)*np.log(1-p)))

def categorical_cross_entropy(p, y):
 return -np.mean(np.log(p[np.arange(len(y)),y]))


# Converting NumPy objects to tensors is baked into PyTorch’s core data structures. you can easily switch back and forth between torch.Tensor objects and numpy.array objects using torch.from_numpy() and Tensor.numpy() methods.
import torch
import numpy as np

x = np.array([[2., 4., 6.]])
y = np.array([[1.], [3.], [5.]])

m = torch.mul(torch.from_numpy(x), torch.from_numpy(y))

m.numpy() 
# exact same method for both PyTorch and TensorFlow whereas DataFrame.to_numpy()



# With PyTorch, requires_grad=True parameter signals to torch.autograd engine that every operation on them should be tracked. (with TensorFlow we need the tf.GradientTape API)
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
L = 3*a**3 - b**2
# We can call .backward() on the loss function (L) of a and b, autograd calculates gradients of the L w.r.t parameters and store them in the respective's tensors' .grad attribute. For example,
external_grad = torch.tensor([1., 1.])
L.backward(gradient=external_grad)
# the gradient parameter specifies the gradient of the function being differentiated w.r.t. self. This argument can be omitted if self is a scalar. here we have a and b.
print(a.grad); print(9*a**2)
print(b.grad); print(-2*b)



import torch

# Import pprint, module we use for making our print statements prettier
import pprint
pp = pprint.PrettyPrinter()

# Create the inputs
input = torch.ones(2,3,4)

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None) y = xA^T + b.

model = torch.nn.Linear(in_features=4, out_features=2) 
# the weights and bias are initalized randomly from U(-sqrt{k}, sqrt{k}) where k=1/ in_features. The size of the weight matrix is out_features x in_features, and the size of the bias vector is out_features.

linear_output = model(input)
linear_output.shape



######### Linear Regression #########
# from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import pandas as pd


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define a function to generate noisy data
def synthesize_data(w, b, sample_size):
  """ Generate y = xW^T + bias + noise """
  X = torch.normal(0, 1, (sample_size, len(w)))
  y = torch.matmul(X, w) + b# add noise
  y += torch.normal(0, 0.01, y.shape)
  
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2., -3.])
true_b = 4.
features, labels = synthesize_data(true_w, true_b, 1000)

def load_data(data_arrays, batch_size, shuffle=True):
 """
 Construct a PyTorch data iterator.
 torch.utils.data.TensorDataset(*tensors) wraps tensors (samples and their corresponding labels). Each sample will be retrieved by indexing tensors along the first dimension.
 torch.utils.data.DataLoader() provides an iterable over the given dataset.
 """
 dataset = TensorDataset(*data_arrays)
 return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
 
batch_size = 10
data_iter = load_data((features, labels), batch_size)
# next(iter(data_iter))
len(data_iter) # sample_size / batch_size = 1000 / 10 = 100

# Create a single layer feed-forward network with 2 inputs and 1 outputs.
model = nn.Linear(2, 1).to(device)
# define a loss function: mean squared error
criterion = nn.MSELoss()
# define a optimization method: stochastic gradient descent 
lr=0.03
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Initialize model params
# the default weights and bias are initalized randomly from U(-sqrt{k}, sqrt{k}) where k=1/ in_features.
model.weight.data.normal_(0, 0.01)
model.bias.data.fill_(0)

# train for n epochs, iteratively using minibatch of the size defined by batch_size vriable
num_epochs = 5
losslist = []
epochlist = []

# When you perform backpropagation, the gradients of the loss with respect to the model's parameters are calculated and stored. If you don't zero out the gradients before the next iteration, the gradients from the previous iteration will be added to the current gradients, leading to incorrect updates.

for epoch in range(num_epochs):
 for X, y in data_iter:
  
  X, y = X.to(device), y.to(device)
  # forward pass
  y_out = model(X)
  l_t = criterion(y_out, y)
  # zero out the gradients 
  optimizer.zero_grad() 

  # backpropagation
  l_t.backward()
  # Update the parameters
  optimizer.step()

  yv_out = model(features)
  l_v = criterion(yv_out, labels)
  print(f'epoch {epoch + 1}, loss {l_v:f}')

  losslist.append(l_v.item())
  epochlist.append(epoch)

result_df = pd.DataFrame({'epoch':epochlist, 'loss':losslist})

# Results
w = model.weight#.tolist()
print('Error in estimating weights:', true_w - w.reshape(true_w.shape))
b = model.bias#.item()
print('Error in estimating bias:', true_b - b)

# Tensor.tolist() returns the tensor as a (nested) list. For scalars, a standard Python number is returned, just like with Tensors.item(). Tensors are automatically moved to the CPU first if necessary. tensor.data attribute was previously used to access the underlying storage of a tensor. However, it's now considered deprecated. Directly modifying tensor.data is generally discouraged as it can lead to unexpected behavior in PyTorch's autograd system.

summary(model) # imilar to Tensorflow's model.summary()


import torch
import torch.nn as nn
import numpy as np

# Setup - data preparation
# Define a function to generate noisy data
def synthesize_data(w, b, sample_size):
  """ Generate y = xW^T + bias + noise """
  X = torch.normal(10, 3, (sample_size, len(w)))
  y = torch.matmul(X, w) + b# add noise
  y += torch.normal(0, 0.01, y.shape)
  
  return X, y.reshape((-1, 1))

def norm(x):
    """ normalize the original data values """
    return (x - np.mean(x)) / np.std(x)

def load_data(tensors, batch_size, is_train=True):
   """ Construct a PyTorch data iterator."""
   dataset = torch.utils.data.TensorDataset(*tensors)
   return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
 
true_w = torch.tensor([2., -3.])
true_b = 4.

train_size = 0.8

X, y = synthesize_data(true_w, true_b, 1000)
size = int(X.shape[-2]*train_size)
index = np.random.choice(X.shape[-2], size=size, replace=False) 

# Prepare the traing set. Note the synthetic data are torch.Tensors. Here it is transform into NumpyArray first for norm operation then reverse back to Tensor.
X_train = torch.from_numpy(norm(X[index].numpy()))
y_train = y[index]
# Prepare the test set.
X_test = torch.from_numpy(norm(np.delete(X, index, axis=0).numpy()))
y_test = np.delete(y, index, axis=0)

# Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
data_iter = load_data((X_train, y_train), batch_size)



# Use GPU when available, the default is CPU 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Step 1: Create model class
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out

# Step 2: Instantiate model class
input_dim = X_train.shape[-1]
output_dim = y_train.shape[-1]
model = LinearRegressionModel(input_dim, output_dim)
model.to(device)

# Step 3: Instantiate Loss class and Optimizer class
criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step 4: Train the model
batch_size = 10
epochs = 100

# loss_list = []
# epoch_list = []
for epoch in range(epochs):
    epoch += 1 # Logging starts at 1 instead of 0
    for X, y in data_iter:
       X = X.to(device)
       y = y.to(device)
       
       optimizer.zero_grad() # Clear gradients w.r.t. parameters
       outputs = model(X) # Forward to get output
       loss = criterion(outputs, y) # Calculate Loss

       # loss_list.append(loss.item())
       # epoch_list.append(epoch)
       print('epoch {}, loss {}'.format(epoch, loss.item())) # Logging
       
       loss.backward() # Getting gradients w.r.t. parameters
       optimizer.step() # Updating parameters



###########

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
import pandas as pd


# It is necessary to have both the model, and the data on the same device, either CPU or GPU, for the model to process data.

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (the minibatch dimension (at dim=0) is maintained).
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # apply a linear transformation on the input using its stored weights and biases
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

"""
Inside the training loop, optimization happens in three steps:

optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

Call loss.backward() to backpropagate the prediction loss. PyTorch deposits the gradients of the loss w.r.t. each parameter.

Call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learning_rate = 1e-3
batch_size = 64
epochs = 10

# Initialize model, and move it to the device    
model = NeuralNetwork()#.to(device)

print(summary(model))

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
   print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
  print(var_name, "\t", optimizer.state_dict()[var_name])




# PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.
modelX = NeuralNetwork()
# Using weights_only=True is considered a best practice when loading weights.
modelX.load_state_dict(torch.load('model_weights.pth', weights_only=True))
modelX.to(device)
# # Make sure to call input = input.to(device) on any input tensors that you feed to the model

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. If you wish to resuming training, call model.train() to set these layers to training mode.
modelX.eval()





X = torch.rand(1, 28, 28, device=device)
logits = model(X)

pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted class: {y_pred}")
# Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.

# 3 images of size 28x28
input_image = torch.rand(3,28,28)
flatten = nn.Flatten()
flat_image = flatten(input_image)
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
hidden1 = nn.ReLU()(hidden1)

