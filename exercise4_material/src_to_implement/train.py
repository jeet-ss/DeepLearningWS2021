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
tt, vt = train_test_split(data_files, train_size=0.2, test_size=0.8,  random_state=1 )
train_data, validation_data = train_test_split(tt, train_size=0.8, test_size=0.2, random_state=1)
train_data = tt
validation_data = tt
# TODO: remove
print("train_data_batches", train_data.__len__(), validation_data.__len__())

# Hyper parameters
learning_rate = 0.01
momentum = 0.9
early_stopping_patience = 100
epochs_num = 1000
batch_size = 50

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_batches = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=batch_size, shuffle=True)
val_batches = t.utils.data.DataLoader(ChallengeDataset(validation_data, 'val'), batch_size=batch_size, shuffle=True)

# create an instance of our ResNet model
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_function = t.nn.BCELoss()
# set up the optimizer (see t.optim)
#optimizer = t.optim.Adam(resnet.parameters(), lr=learning_rate)
optimizer = t.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=momentum)
# create an object of type Trainer and set its early stopping criterion
cuda_check = t.cuda.is_available()
# TODO: remove
print("cuda:", cuda_check)  #
trainer_object = Trainer(model=resnet, crit=loss_function, optim=optimizer, train_dl=train_batches, val_test_dl=val_batches, cuda=cuda_check, early_stopping_patience=early_stopping_patience)

# go, go, go... call fit on trainer
res = trainer_object.fit(epochs=epochs_num)
model_path = "resnet_model_trained.pt"
t.save(resnet, model_path)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])),  res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')