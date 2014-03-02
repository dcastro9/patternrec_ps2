from Dataset import Dataset
from WTA_Hasher import WTAHasher

ds_orig = Dataset("../datasets/alcohol/alcoholism_training.csv")
ds_norm = Dataset("../datasets/alcohol/alcoholism_training.csv",
                  normalize=True)
ds_norm_weigh = Dataset("../datasets/alcohol/alcoholism_training.csv",
	                    normalize=True, weights=[1,1,1,1,2,3])
wta_hasher1 = WTAHasher(3, 5, ds_norm)
wta_hasher1.hashDataset("../datasets/alcohol/alcoholism_training_wta_norm.csv")

wta_hasher2 = WTAHasher(3, 5, ds_norm_weigh)
wta_hasher2.hashDataset(
    "../datasets/alcohol/alcoholism_training_wta_norm_weigh.csv")

ds_wta_norm = Dataset("../datasets/alcohol/alcoholism_training_wta_norm.csv")
ds_wta_norm_weigh = Dataset(
    "../datasets/alcohol/alcoholism_training_wta_norm_weigh.csv")

alcohol_datasets = [ds_orig, # Original dataset.
                    ds_norm, # Normalized from 0 to 1.
                    ds_norm_weigh, # Normalized + manually determined weights.
                    ds_wta_norm, # Normalized + WTA Hash (rank correlation)
                    ds_wta_norm_weigh] # Normalized + Weighed + WTA Hash
