import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras



class MultiModalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, points_data, images_data, labels_data, batch_size, shuffle=True):
        self.points_data = points_data
        self.images_data = images_data
        self.labels_data = labels_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.labels_data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.labels_data) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        batch_points = self.points_data[batch_indices]
        batch_images = self.images_data[batch_indices]
        batch_labels = self.labels_data[batch_indices]

        # Perform any preprocessing or data augmentation as needed
        # For example, you can normalize images or apply transformations to points

        return [batch_points, batch_images], batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)