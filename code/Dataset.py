# Copyright 2014, MIT License.
# Author: Daniel Castro <dcastro9@gatech.edu>

import csv
import itertools
import math
import numpy as np
from random import sample
from random import shuffle

class Dataset(object):
    """ Represents a dataset. Can divide the dataset into 'k' classes,
    and provide training & testing data.

    Attributes:
        file_path: Location of the CSV data file.
        whiten: Subtracts the mean from each descriptor, choose whiten OR
                normalize, not both.
        normalize: Defaults to false, can be True, will normalize from 0 to 1
        weights: If you want to custom weigh the normalization from 0 to n
                 pass in a weight vector of the length of the number of
                 features (without counting the classification) that has the
                 'n' value you want to normalize to. For WTA Hash, this can be
                 useful in giving certain features more weight than others.
        name: Name your dataset.

    """

    def __init__(self, file_path, whiten=False, normalize=False, weights=None,
                 name=''):
        """Creates a Dataset object.
        """
        self._data = []
        self._classes = []
        self._classified_data = []
        self._norm_vals = []
        self._name = name

        with open(file_path, 'rb') as file_input:
            file_reader = csv.reader(file_input, delimiter=',')
            for data_point in file_reader:
                for val in range(len(data_point)):
                    data_point[val] = float(data_point[val])
                    if (whiten and val != len(data_point) - 1):
                        if (len(self._norm_vals) < len(data_point) - 1):
                            self._norm_vals.append(data_point[val])
                        else:
                            self._norm_vals[val] += data_point[val]
                    elif (normalize and val != len(data_point) - 1):
                        if (len(self._norm_vals) < len(data_point) - 1):
                            # Sets the minimum and max to the first value.
                            self._norm_vals.append(
                                [data_point[val], data_point[val]])
                        else:
                            if self._norm_vals[val][0] > data_point[val]:
                                self._norm_vals[val][0] = data_point[val]
                            elif self._norm_vals[val][1] < data_point[val]:
                                self._norm_vals[val][1] = data_point[val]
                self._data.append(data_point)
                if int(data_point[-1]) not in self._classes:
                    self._classes.append(int(data_point[-1]))
        # Create an 'n' number of arrays, one per class.
        for val in range(len(self._classes)):
            self._classified_data.append([])
        # Normalize the data from 0 to 1.
        if whiten:
            for val in range(len(self._norm_vals)):
                self._norm_vals[val] /= len(self._data)
            for data_point in self._data:
                for val in range(len(data_point) - 1):
                    data_point[val] /= self._norm_vals[val]
        elif normalize:
            if weights and len(weights) != len(data_point) - 1:
                raise ValueError("Weights is not the correct length.")
            elif not weights:
                weights = []
                for val in range(len(self._data[0]) - 1):
                    weights.append(1)
            for data_point in self._data:
                for val in range(len(data_point) - 1):
                    minVal = self._norm_vals[val][0]
                    maxVal = self._norm_vals[val][1]
                    data_point[val] = (data_point[val] - minVal) / \
                        ((maxVal - minVal) / weights[val])
        # Append data point to its classified class.
        for data_point in self._data:
            self._classified_data[self._classes.index(data_point[-1])].append(data_point)

    def getDataForClass(self, class_index):
        """ Obtains all the data for a given class.

        Attributes:
            class_index: Index of the class you want.
        """
        return np.array(self._classified_data[class_index])

    def getRandomDataForClass(self, num_samples, class_index):
        """ Obtains a random sample of data for a given class.

        Attributes:
            num_samples: Number of random samples you want.
            class_index: Index of the class you want.
        """
        if num_samples > len(self._classified_data[class_index]):
            raise ValueError("Not enough data samples.")
        return np.array(sample(
            self._classified_data[class_index], num_samples))

    def getRandomPercentOfData(self, training_percent):
        """ Returns both training and test data for a random percentage of the
        dataset.

        Attributes:
            training_percent: Value between 0-1 that represents the percent of
                              data you want. Will return 1 - training_percent
                              percent as the test data (the rest).
        """
        return Dataset.getRandomPercent(self._data, training_percent)

    @staticmethod
    def getRandomPercent(data, percent):
        indices = range(len(data))
        shuffle(indices)
        separator = int(len(data)*percent)

        training_data = []
        test_data = []
        for index in indices[:separator]:
            training_data.append(data[index])
        for index in indices[separator:]:
            test_data.append(data[index])

        return training_data, test_data

    @staticmethod
    def getkPartitions(data, k):
        indices = range(len(data))
        shuffle(indices)
        partitions = []

        previous_step = 0
        for step in range(0, len(data), len(data)/k):
            partitions.append(data[previous_step:step])

 
    def kFoldCrossValidation(self, k):
        # Find the smallest class.
        size = None
        for data_class in self._classified_data:
            if size == None:
                size = len(data_class)
            elif size > len(data_class):
                size = len(data_class)

        # Determine ideal bucket size.
        bucket_size = size/(k)
        buckets = []
        for val in range(k):
            train = []
            test = []
            for cur_class in self.classes:
                train.append(np.array( \
                    self._classified_data[self.classes.index(cur_class)] \
                    [val*bucket_size:(val+1)*bucket_size]))

                left_side = self._classified_data \
                    [self.classes.index(cur_class)][0:val*bucket_size]
                right_side = self._classified_data \
                    [self.classes.index(cur_class)][(val+1)*bucket_size:]
                
                if len(left_side) > 0:
                    if len(test) == 0:
                        test = left_side
                    else:
                        test = np.append(test, left_side, axis=0)
                if len(right_side) > 0:
                    if len(test) == 0:
                        test = right_side
                    else:
                        test = np.append(test, right_side, axis=0)
            buckets.append([train, test])
        return buckets
                
    @property
    def dimension(self):
        return len(self._data[0]) - 1

    @property
    def classes(self):
        return self._classes

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name