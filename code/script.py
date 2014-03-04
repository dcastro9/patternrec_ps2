from Dataset import Dataset
from WTA_Hasher import WTAHasher
from kNN_Classifier import kNNClassifier
import numpy as np

ds_train_dir = "../datasets/alcohol/alcoholism_training.csv"
ds_test_dir = "../datasets/alcohol/alcoholism_test.csv"
num_k_values = 21

ds_orig = Dataset(ds_train_dir)
ds_norm = Dataset(ds_train_dir, normalize=True)
ds_norm_weigh = Dataset(ds_train_dir, normalize=True, weights=[1,1,1,1,3,4])
ds_whiten = Dataset(ds_train_dir, whiten=True)

ds_orig_t = Dataset(ds_test_dir)
ds_norm_t = Dataset(ds_test_dir, normalize=True)
ds_norm_weigh_t = Dataset(ds_test_dir, normalize=True, weights=[1,1,1,1,1,2])
ds_whiten_t = Dataset(ds_test_dir, whiten=True)

alcohol_datasets = [[ds_orig, ds_orig_t],
                    [ds_norm, ds_norm_t],
                    [ds_norm_weigh, ds_norm_weigh_t],
                    [ds_whiten, ds_whiten_t]]

for ds in alcohol_datasets:
    train_data_all = ds[0].data
    test_data = ds[1].data

    # Randomly get 20%, 50%, and 80% of the data.
    train_accuracy = [np.zeros(num_k_values),
                      np.zeros(num_k_values),
                      np.zeros(num_k_values),
                      np.zeros(num_k_values)]

    for iter in range(5):
        train_data_20, t = Dataset.getRandomPercent(train_data_all, 0.2)
        train_data_50, t = Dataset.getRandomPercent(train_data_all, 0.5)
        train_data_80, t = Dataset.getRandomPercent(train_data_all, 0.8)

        all_training_data = [train_data_20,
                             train_data_50,
                             train_data_80,
                             train_data_all]
        
        k_values = range(1,num_k_values*2,2)
        for val in range(len(all_training_data)):
            for k in k_values:
                knn = kNNClassifier(all_training_data[val], k)
                out = knn.test(test_data)
                train_accuracy[val][k/2] += out

    print "Average Accuracy Array for 20%, 50%, and 80%, and k from 1 to 41."
    for acc in train_accuracy:
        acc /= 5
        print acc
    print "\n"