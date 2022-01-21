import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data_files = pd.read_csv('data.csv', sep=';')
train_data, test_data = train_test_split(data_files, train_size=0.7, test_size=0.3, random_state=1, shuffle=True)
print("train_data_batches", train_data.__len__(), test_data.__len__())

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
batch_size = 50
# train_batches = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=batch_size, shuffle=True)
val_batches = t.utils.data.DataLoader(ChallengeDataset(test_data, 'val'), batch_size=batch_size, shuffle=True)

# create an instance of our ResNet model
model_path = 'resnet_model_trained.pt'
resnet_trained = t.load(model_path)

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_function = t.nn.BCELoss()
# set up the optimizer (see t.optim)
#optimizer = t.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)   # Hyper parameters
# create an object of type Trainer and set its early stopping criterion
early_stopping_patience = 5  # Hyper parameters
cuda_check = t.cuda.is_available()
print("cuda:", cuda_check)  #
tester_object = Trainer(model=resnet_trained, crit=loss_function, optim=None, train_dl=None, val_test_dl=val_batches, cuda=cuda_check, early_stopping_patience=early_stopping_patience)

# go, go, go... call fit on trainer
res = tester_object.test_loop(epochs=2)

# plot the results
plt.plot(np.arange(len(res)), res, label='test loss')
#plt.plot(np.arange(len(res[0])), res[0], label='train loss')
#plt.plot(np.arange(len(res[1])),  res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('test_losses.png')
