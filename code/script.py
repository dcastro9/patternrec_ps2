from Dataset import Dataset
from WTA_Hasher import WTAHasher
from kNN_Classifier import kNNClassifier
import numpy as np
import matplotlib.pyplot as plt

ds_train_dir = "../datasets/alcohol/alcoholism_training.csv"
ds_test_dir = "../datasets/alcohol/alcoholism_test.csv"
num_k_values = 21
weights = [1,1,1,1,1,3]

ds_orig = Dataset(ds_train_dir, name='Original Data')
ds_norm = Dataset(ds_train_dir, normalize=True, name='Normalized Data')
ds_norm_weigh = Dataset(ds_train_dir, normalize=True, weights=weights,
                        name='Norm & Weighted Data')
ds_whiten = Dataset(ds_train_dir, whiten=True, name='Whitened Data')

ds_orig_t = Dataset(ds_test_dir)
ds_norm_t = Dataset(ds_test_dir, normalize=True)
ds_norm_weigh_t = Dataset(ds_test_dir, normalize=True, weights=weights)
ds_whiten_t = Dataset(ds_test_dir, whiten=True)

alcohol_datasets = [[ds_orig, ds_orig_t],
                    [ds_norm, ds_norm_t],
                    [ds_norm_weigh, ds_norm_weigh_t],
                    [ds_whiten, ds_whiten_t]]

k_values = range(1,num_k_values*2,2)
color=['red','blue','green','black']

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
        for val in range(len(all_training_data)):
            for k in k_values:
                knn = kNNClassifier(all_training_data[val], k)
                out = knn.test(test_data)
                train_accuracy[val][k/2] += out

    plt.xlabel('k Values')
    plt.ylabel('Accuracy')
    plt.title(ds[0].name)
    out_f = open(ds[0].name + ".txt", 'w')
    for cnt in range(len(train_accuracy)):
        train_accuracy[cnt] /= 5
        plt.plot(k_values, train_accuracy[cnt], color=color[cnt])
        out_f.write(train_accuracy[cnt] + "\n")
    out_f.close()
    plt.savefig(ds[0].name + ".pdf")
    plt.clf()
    plt.cla()