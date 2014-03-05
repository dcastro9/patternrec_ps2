from Dataset import Dataset
from WTA_Hasher import WTAHasher
from kNN_Classifier import kNNClassifier
import numpy as np
import matplotlib.pyplot as plt
import copy

ds_train_dir = "../datasets/pima_indians/data.csv"
results_dir = "../final_results/pima_indians/"
num_k_values = 10

ds_orig = Dataset(ds_train_dir, name='Original Data')
ds_whiten = Dataset(ds_train_dir, whiten=True, name='Whitened Data')

alcohol_datasets = [ds_orig, ds_whiten]

k_values = range(1,num_k_values*2,2)
color=['red','blue','green','black']
labels=['20%', '50%', '80%', '100%']
folds=['2-fold', '5-fold', 'N-fold']

for ds in alcohol_datasets:
    train_data_all, test_data = ds.getRandomPercentOfData(0.8)

    # Accuracy for get 20%, 50%, 80% and 100% of the data.
    # Each subset will have 
    train_accuracy = [[np.zeros(num_k_values), np.zeros(num_k_values), np.zeros(num_k_values)],
                      [np.zeros(num_k_values), np.zeros(num_k_values), np.zeros(num_k_values)],
                      [np.zeros(num_k_values), np.zeros(num_k_values), np.zeros(num_k_values)],
                      [np.zeros(num_k_values), np.zeros(num_k_values), np.zeros(num_k_values)]]
    best_k_and_ds = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for it in range(5):
        train_data_20, t = Dataset.getRandomPercent(train_data_all, 0.2)
        train_data_50, t = Dataset.getRandomPercent(train_data_all, 0.5)
        train_data_80, t = Dataset.getRandomPercent(train_data_all, 0.8)
        all_training_data = [train_data_20,
                             train_data_50,
                             train_data_80,
                             train_data_all]
        # Only run on train_data_all once.
        if it > 0:
            all_training_data = all_training_data[:-1]
        for val in range(len(all_training_data)):
            for k in k_values:
                print str(it) + ": Training on: " + labels[val] + "for k value: " + str(k) + " for " + ds.name
                # Do 2-5-N Fold Cross Validation.
                cv_2 = Dataset.getkPartitions(all_training_data[val], 2)
                cv_5 = Dataset.getkPartitions(all_training_data[val], 5)
                cv_n = Dataset.getkPartitions(all_training_data[val],
                                              len(all_training_data[val]))
                cvs = [cv_2, cv_5, cv_n]
                cross_val_accuracy = [0, 0, 0]
                for cv_c in range(len(cvs)):
                    # Does f-Fold cross validation.
                    accuracy = 0
                    for fold in range(len(cvs[cv_c])):
                        td = copy.deepcopy(cvs[cv_c]) # Copy the cross validation dataset.
                        del td[fold] # Delete the item we're using for testing.
                        td_reshaped = []
                        for elem in td:
                            for item in elem:
                                td_reshaped.append(item)
                        knn = kNNClassifier(td_reshaped, k) # Initialize kNN.
                        accuracy += knn.test(cvs[cv_c][fold]) # Test.
                    accuracy /= len(cvs[cv_c])

                    if best_k_and_ds[val][cv_c] == 0:
                        best_k_and_ds[val][cv_c] = [k, td_reshaped, accuracy]
                    elif best_k_and_ds[val][cv_c][2] < accuracy:
                        best_k_and_ds[val][cv_c] = [k, td_reshaped, accuracy]
                    train_accuracy[val][cv_c][k/2] += accuracy

    # Write results to file.
    out_f = open(results_dir + ds.name + ".txt", 'w')
    for cnt in range(len(train_accuracy)):
        # Setup plot.
        plt.xlabel('k Values')
        plt.ylabel('Accuracy')
        plt.title(ds.name)
        average = True
        if cnt == len(train_accuracy) - 1:
            average = False
        for fold in range(len(train_accuracy[cnt])):
            if (average):
                train_accuracy[cnt][fold] /= 5
            plt.plot(k_values, train_accuracy[cnt][fold], color=color[fold],
                label=folds[fold])
            out_f.write(labels[cnt] + ":" + folds[fold] + ":" +
                        str(train_accuracy[cnt][fold]) + "\n")
        # Save plot.
        plt.legend()
        plt.savefig(results_dir + ds.name + labels[cnt] + ".pdf")
        plt.clf()
        plt.cla()

    # Now we test with the original test data provided.
    out_f.write("\n\n Testing for best k & DS for:" + ds.name +"\n")

    for val in range(len(best_k_and_ds)):
        for fold in range(len(best_k_and_ds[val])):
            knn = kNNClassifier(best_k_and_ds[val][fold][1],
                                best_k_and_ds[val][fold][0]) # Initialize kNN.
            out = knn.test(test_data) # Test.
            out_f.write(labels[val] + " with k:" + 
                str(best_k_and_ds[val][fold][0]) + " at " + folds[fold] +
                " accuracy:" + str(out) + "\n")
    # Close file.
    out_f.close()