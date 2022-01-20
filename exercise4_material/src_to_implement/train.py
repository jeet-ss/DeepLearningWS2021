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
tt, vt = train_test_split(data_files, train_size=0.99, test_size=0.01, random_state=1 )
train_data, validation_data = train_test_split(vt, train_size=0.8, test_size=0.2, random_state=1)
print("train_data", train_data.__len__(), validation_data.__len__())

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_batches = t.utils.data.DataLoader(ChallengeDataset(train_data, 'train'), batch_size=2, shuffle=True)
val_batches = t.utils.data.DataLoader(ChallengeDataset(validation_data, 'val'), batch_size=2, shuffle=True)

# create an instance of our ResNet model
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
loss_function = t.nn.BCELoss()
optimizer = t.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9)   # Hyper parameters
early_stopping_patience = 2  # Hyper parameters
trainer_object = Trainer(model=resnet, crit=loss_function, optim=optimizer, train_dl=train_batches, val_test_dl=val_batches, cuda=False, early_stopping_patience=early_stopping_patience)

# go, go, go... call fit on trainer
res = trainer_object.fit(epochs=3)
print(len(res[0]))
# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')