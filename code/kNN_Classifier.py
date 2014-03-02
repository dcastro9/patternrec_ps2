# Copyright 2014, MIT License.
# Author: Daniel Castro <dcastro9@gatech.edu>

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

    def test(testing_data):
    	for data_point in testing_data:
    		# Compute distance to each training data point.

    		# Choose minimum k distances.

    		# Choose most common class in k distances.