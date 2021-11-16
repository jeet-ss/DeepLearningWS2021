import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize #for image resizing
import itertools #for slicing a dict
import random #for shuffling

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        #read json data from file
        with open(label_path, 'r') as file:
            json_data = json.load(file)

        self.json_data = json_data

        #define epoch to increase later
        self.epoch = 0
        self.index = 0


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #implement next method

        #arrays for output
        batch_images = []
        image_labels = []

        #shuffle the dict and get back a dict
        if self.shuffle:
            l=list(self.json_data.items())
            random.shuffle(l)
            self.json_data=dict(l)

        #slice the dictionary as per batch size
        batch_data = dict(itertools.islice(self.json_data.items(), self.index, self.index + self.batch_size)) #returns small dict if less elements present

        #update epoch
        if len(batch_data) < self.batch_size: #
            self.epoch += 1
            self.index = self.batch_size - len(batch_data) #calculate the overflow
            # append the values from the starting to the batch
            batch_data.update(dict(itertools.islice(self.json_data.items(), self.index)))
        else:
            self.index += self.batch_size #increase array pointer by the batch size

        #loop through the images based on batch size
        for key in batch_data:
            #load image file from the file path
            image = np.load(self.file_path + key + '.npy')

            #resizing the image if it deos not match the size
            if image.shape != (self.image_size[0], self.image_size[1], self.image_size[2]):
                image = resize(image, (self.image_size[0], self.image_size[1]))

            #rotation and mirroring
            image = self.augment(image)

            #put the augmented images in the array
            batch_images.append(image)
            image_labels.append(batch_data[key])

        #Update the output arrays
        images = np.array(batch_images)
        labels = np.array(image_labels)

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # implement augmentation function

        #mirroring
        if self.mirroring:
            num = np.random.choice([1,2,3])
            if num == 1:
                #H-flip
                img = np.flip(img, 1)
            elif num == 2:
                #V-flip
                img = np.flip(img, 0)
            elif num == 3:
                #H & V
                img = np.flip(img, (0,1))
            else:
                img = img #none

        #rotation
        if self.rotation:
            random_float = np.random.sample()
            if random_float <= 0.25:
                #rot by 90
                img = np.rot90(img)
            elif np.logical_and(random_float>0.25, random_float<=0.5):
                #rot by 180
                img = np.rot90(img, 2)
            elif np.logical_and(random_float>0.5, random_float<=0.75):
                #rot by 270
                img = np.rot90(img, 3)
            else:
                img = img #none

        return img

    def current_epoch(self):
        #Treturn the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        # implement class name function
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # implement show method

        #call next method
        images, labels = self.next()

        #find out the grid size
        num = np.int_(np.ceil(np.sqrt(self.batch_size)))
        #show the images
        plt.figure()
        for x in range(self.batch_size):
            #take out the label of the image
            label = self.class_name(labels[x])
            #plot the images
            plt.subplot(num, num, x+1) #x+1 because subplot takes values from 1 and not 0
            plt.title(label) #show tilte
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[x])

        plt.show()



