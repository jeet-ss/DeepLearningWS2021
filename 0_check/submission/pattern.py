import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        #This is the constructor
        if (resolution % (2*tile_size)!=0):
            print("resolution and tile size should match")
        else:
            self.resolution = resolution
            self.tile_size = tile_size
            self.output = np.empty(0)

    def draw(self):
        #the draw function
        zeros = np.zeros((self.tile_size, self.tile_size), dtype=int) #black tile
        ones = np.ones((self.tile_size, self.tile_size), dtype=int) #white tile
        #define the no of repetition of each cell
        reps = int(self.resolution/(2*self.tile_size))
        #create black and white tile
        x = np.concatenate((zeros, ones), axis=1)
        #create reverse tile or white then black
        y = np.concatenate((ones, zeros), axis=1)
        #create row as chain of repeated tiles
        x1 = np.tile(x, reps)
        y1 = np.tile(y, reps)
        #concatenate two rows
        two_rows = np.concatenate((x1,y1), axis=0)
        #repeat the two rows
        checker_pattern = np.tile(two_rows, (reps, 1))
        #Outro
        self.output = np.copy(checker_pattern)
        return np.copy(checker_pattern)


    def show(self):
        #the show or print function
        fig, ax = plt.subplots()
        ax.imshow(self.draw(), cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.show()


#new class
class Circle:
    def __init__(self, resolution, radius, postion):
        #constructor for the class
        self.resolution = resolution
        self.radius = radius
        self.position = postion
        self.output = np.empty(0)

    def draw(self):
        #define meshgrid based on resolution
        x,y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        # black picture
        picture = np.zeros((self.resolution, self.resolution)) #black picture
        #compute distance of meshgrid from the center position
        circle = np.sqrt((x-self.position[0])**2 + (y-self.position[1])**2)
        #threshold black picture based on radius
        picture[circle<=self.radius ] = 1
        #output
        self.output= picture
        return np.copy(picture)

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.draw(), cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()



#new class
class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.empty(0)

    def draw(self):
        line = np.linspace(0, 1 , self.resolution)
        x, y = np.meshgrid(line, line)
        #empty matrix for spectrum
        spectrum = np.zeros((self.resolution, self.resolution, 3))
        #define the channels
        spectrum[:,:,0] = x
        spectrum[:,:,1] = y
        spectrum[:, : ,2] = 1-x

        #copy output
        self.output=np.copy(spectrum)
        return np.copy(spectrum)

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.draw(), cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()