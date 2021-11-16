
import json
import numpy as np

if __name__ == "__main__":
    label_path = './Labels.json'
    with open(label_path, 'r') as file:
        json_data = json.load(file)
    keys_temp = []
    for key in json_data.keys():
        keys_temp.append(key)
    dict_keys = np.array(keys_temp)
    print(dict_keys[0])

    # extract keys and values form json data
    # keys or the images names
    keys_temp = []
    for key in self.json_data.keys():
        keys_temp.append(key)
    self.dict_keys = np.array(keys_temp)
    # values
    val_temp = []
    for key in self.json_data.keys():
        val_temp.append(self.json_data[key])
    self.dict_values = np.array(val_temp)

    # loop through the images based on batch size
    for key in self.dict_keys[self.start:self.end]:
        # load image file from the file path
        image = np.load(self.file_path + key + '.npy')

        # resizing the image if it deos not match the size
        if image.shape != (self.image_size[0], self.image_size[1], self.image_size[2]):
            image = resize(image, (self.image_size[0], self.image_size[1]))

        # image rotation
        if self.rotation:
            image = self.augment(image)

        # mirroring image
        if self.mirroring:
            image = self.augment(image)

        # put the augmented images in the array
        batch_images.append(image)

    # loop through the labels array
    for value in self.dict_values[self.start:self.start + self.batch_size]:
        image_labels.append(value)

    # move the array start forward
    self.start += self.batch_size
    self.end += self.batch_size

    # roll the array to the element after the batch ends
    # self.dict_keys = np.roll(self.dict_keys, -self.batch_size)
    # self.dict_values = np.roll(self.dict_values, -self.batch_size)

    # Update the output arrays
    images = np.array(batch_images)
    labels = np.array(image_labels)

