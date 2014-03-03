# Copyright 2014, MIT License.
# Author: Daniel Castro <dcastro9@gatech.edu>

from collections import Counter
import numpy as np

class kNNClassifier(object):
    """ Represents a k-Nearest Neighbor classifier. It can generate a
    classifier based on training data and then attempt to classify the test
    data accordingly. 

    Attributes:
       training_data: Numpy array of training data.
       k: Number of nearest neighbors.
    """

    DISTANCE_MEASURES = {0:'Eucledian', 1:'Cayley'}

    def __init__(self, training_data, k, distance=0):
    	self._training_data = training_data
    	self._k_value = k
        self._distance = distance

    def test(testing_data):
        correct = 0
        incorrect = 0
    	for test_point in testing_data:
            # Compute k minimum distance points out of training data.
            minimum_distances = []
            for train_point in self._training_data:
                dist = self.__distance(test_point[:-1], train_point[:-1])
                if len(minimum_distances) < self._k_value:
                    minimum_distances.append([dist, train_point[-1]])
                else:
                    l_distance = minimum_distances[0][0]
                    l_index = 0
                    for ind in range(len(minimum_distances)):
                        if minimum_distances[ind][0] > l_distance:
                            l_distance = minimum_distances[ind][0]
                            l_index = ind
                    if dist < l_distance:
                        minimum_distances[l_index] = [dist, train_point[-1]]

    		# Choose most common class in k distances.
            classes = []
            for distances in minimum_distances:
                classes.append(distances[1])
            classification = Counter(classes).most_common(1)[0][0]
            if int(test_point[-1]) == int(classification):
                correct += 1
            else:
                incorrect += 1
        return correct / (correct + incorrect)



    def __distance(v1, v2):
        if (self._distance == 0):
            return np.linalg.norm(v1-v2)