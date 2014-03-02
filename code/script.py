from Dataset import Dataset
from WTA_Hasher import WTAHasher

# What if we generated the weights using something like PCA, with the eigen
# values of the matrix to weigh the normalization. May actually give us really
# good results for most, if not all, datasets.

ds = Dataset("../datasets/alcohol/alcoholism_training.csv",
	normalize=True, weights=[1,1,1,1,2,3])

wta_hasher = WTAHasher(3, 5, ds)
wta_hasher.hashDataset("")