import numpy as np
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=False,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
        # self added
        self.epoch_counter = 0

            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        pred = self._model(x)
        # -calculate the loss
        loss = self._crit(pred, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        
        # predict
        pred = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(pred, y)
        # TODO : set pred to 0 or 1
        # return the loss and the predictions
        return loss, pred

    def train_epoch(self):
        # set training mode
        self._model.training = True
        # iterate through the training set
        loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                # transfer the batch to "cuda()" -> the gpu if a gpu is given
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss += self.train_step(x,y)
        # calculate the average loss for the epoch and return it
        avg_loss = loss/self._train_dl.__len__()
        return avg_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # initialize variables
        loss = 0
        batch_pred = t.empty(0)
        batch_labels = t.empty(0)
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for x, y in self._val_test_dl:
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                    batch_pred = batch_pred.cuda()
                    batch_labels = batch_labels.cuda()
                # perform a validation step
                step_loss, pred = self.val_test_step(x, y)
                loss += step_loss
                # save the predictions and the labels for each batch
                batch_pred = t.cat((batch_pred, pred))
                batch_labels = t.cat((batch_labels, y))
        # calculate the average loss
        avg_loss = loss/self._val_test_dl.__len__()
        # calculate average metrics of your choice. You might want to calculate these metrics in designated functions
        # for whole validation
        f1_metric = f1_score(batch_labels.cpu(), batch_pred.cpu() > 0.5, average='weighted')
        # return the loss and print the calculated metrics
        print("f1_score", f1_metric, " at epoch: ", self.epoch_counter)
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        loss_train = np.array([])
        loss_val = np.array([])
        epoch_counter = 0
        min_loss_val = np.Inf
        criteria_counter = 0
        # TODO: remove
        print("cuda inside fit:", self._cuda)
        
        while True:
            # stop by epoch number
            if epoch_counter >= epochs:
                break
            # increment Counter
            epoch_counter += 1
            self.epoch_counter = epoch_counter
            # train for an epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            val_loss = self.val_test()
            # port to cpu as gpu version cannot be converted to numpy
            train_loss = train_loss.cpu()
            val_loss = val_loss.cpu()
            # append the losses to the respective lists
            loss_train = np.append(loss_train, train_loss.detach().numpy())
            loss_val = np.append(loss_val, val_loss.detach().numpy())
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # self.save_checkpoint(epoch_counter)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if val_loss < min_loss_val:
                min_loss_val = val_loss
                criteria_counter = 0
                self.save_checkpoint(epoch_counter)
            else:
                criteria_counter += 1

            if criteria_counter > self._early_stopping_patience:
                print("Early Stopping Criteria activated")
                break
        # return the losses for both training and validation

        return loss_train, loss_val
        
    def test_loop(self, epochs=-1):
        assert epochs > 0
        loss = np.array([])
        epoch_counter = 0
        while True:
            # stop by epoch number
            if epoch_counter >= epochs:
                break
            # increment Counter
            epoch_counter += 1
            self.epoch_counter = epoch_counter
            t_loss = self.val_test()
            t_loss = t_loss.cpu()
            loss = np.append(loss, t_loss.detach().numpy())
        return loss

