# Copyright 2014, MIT License.
# Author: Daniel Castro <dcastro9@gatech.edu>

from random import shuffle
import numpy as np

class WTAHasher(object):
    """ Processes a dataset, and converts it to a rank correlation dataset
    based on a given k, and a random set of permutations.

    Attributes:
        k: Subset of the original vector.
        num_permutations: Number of permutations (this determines the dimensions
       					 of your new data vector).
		dataset: A Dataset object with your current data.
    """

    def __init__(self, k, num_permutations, dataset):
    	self._k_value = k
    	self._num_perm = num_permutations
    	self._dataset = dataset
    	self._permutations = []
    	for val in range(num_permutations):
    		indices = range(self._dataset.dimension)
    		shuffle(indices)
    		self._permutations.append(indices[:k])

    		

    def hashDataset(self, out_file):
    	# Create out_file.
    	for data_point in self._dataset.data:
    		generated_hash = ""
    		for perm in self._permutations:
    			generated_hash += \
    				str(self.__getHashCode(data_point[:-1], perm)) + ","
    		generated_hash = generated_hash[:-1] + "\n"
    		# Write hashed point to file.
    	# Save file.



    def __getHashCode(self, data_point, permutation):
    	# Temporary Array for permuted data point.
    	temp_array = []
    	# Permute data_point.
    	for index in permutation:
    		temp_array.append(data_point[index])

    	# Returns the index of the maximum of the subset of the datapoint.
    	return np.argmax(temp_array)