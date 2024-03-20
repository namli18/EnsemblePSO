# BPSO with RF (Feature selection and Classification)

import numpy as np
import pandas as pd
import pyswarms as ps
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from pyswarms.utils.plotters import plot_cost_history
from sklearn.ensemble import RandomForestClassifier


name= "BPSO+RF"
numberofrun=1
n1 = dt.datetime.now()

def __readData(fileName_input, fileName_label, sep=',') -> tuple:
    data = pd.read_csv(fileName_input, sep=sep)
    label = pd.read_csv(fileName_label, sep=sep)
    data_features = np.array(data)
    data_label = np.array(label)
    return data_features, data_label

X_train, y_train = __readData(
        fileName_input='.../datatset_train_x.csv',
        fileName_label='.../datatset_train_y.csv')

X_test, y_test = __readData(
        fileName_input='.../datatset_test_x.csv',
        fileName_label='.../datatset_test_y.csv')

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

y_train_flatten=y_train.flatten()
y_test_flatten=y_test.flatten()

# One hot encode y values for neural network.
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
numberofclass=y_train.shape[1]
print("X_train.shape[0]",X_train.shape[0])
print("X_train.shape[1]",X_train.shape[1])
print("y",y_train_flatten)


def performance(outputMatrix, Test_Yd_calculated_Class):
    cm = confusion_matrix(outputMatrix, Test_Yd_calculated_Class)
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for {0}'''.format(name))
    accuracy = round(accuracy_score(outputMatrix, Test_Yd_calculated_Class), 3)
    recall = round(recall_score(outputMatrix, Test_Yd_calculated_Class, average="macro"), 3)
    precision = round(precision_score(outputMatrix, Test_Yd_calculated_Class, average="macro"), 3)
    fmeasure = round(f1_score(outputMatrix, Test_Yd_calculated_Class, average="macro"), 3)
    labels = list(range(0, numberofclass))
    outputMatrix_b = label_binarize(outputMatrix, classes=labels)
    Test_Yd_calculated_Class_b = label_binarize(Test_Yd_calculated_Class, classes=labels)
    roc = round(roc_auc_score(outputMatrix_b, Test_Yd_calculated_Class_b, multi_class='ovr', average="macro"), 3)
    my_dict = {"Accuracy": accuracy,
               "Recall": recall,
               "Precision": precision,
               "F-measure": fmeasure,
               "Roc": roc}
    print(my_dict)
    print("")
    print("Confusion Matrix:")
    print(cm)
    # One hot encode y values for neural network.
    from keras.utils import to_categorical
    outputMatrix = to_categorical(outputMatrix, num_classes=numberofclass, dtype="int32")
    Test_Yd_calculated_Class = to_categorical(Test_Yd_calculated_Class, num_classes=numberofclass, dtype="int32")
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(outputMatrix.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(outputMatrix[:, i], Test_Yd_calculated_Class[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(outputMatrix.shape[1])]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(outputMatrix.shape[1]):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= outputMatrix.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)

    colors = cycle(
        ['pink', 'violet', 'lightgreen', 'yellow', 'aqua', 'darkorange', 'cornflowerblue', 'teal', 'lime',
         'lightcoral'])
    for i, color in zip(range(outputMatrix.shape[1]), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for {0}'''.format(name))
    plt.legend(loc="lower right")
    plt.show()
    # draw ROC Curve

# Calculate objective function value for each particles
def f_per_particle(b, alpha):
    # X_subset gives the subset of the features
    if np.count_nonzero(b) == 0:
        X_subset = X_train
    else:
        X_subset = X_train[:,b==1]

    model = RandomForestClassifier(n_estimators = 50, random_state = 42)
    history = model.fit(X_subset, y_train_flatten)
    P = round(accuracy_score(y_train_flatten, np.around(model.predict(X_subset))), 3)

    # j gives the objective function value
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (X_subset.shape[1] / dimensions))
    return j

def f(x, alpha=0.95):
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Create an instance of the classifier
# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 3, 'p':2}

# Call instance of PSO
n_particles=5
dimensions = X_train.shape[1] # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)

n2 = dt.datetime.now()
# Get the selected features from the final positions
X_selected_features_train = X_train[:,pos==1]  # subset
X_selected_features_test = X_test[:,pos==1]  # subset
print("cost",cost)
print("pos",pos)
print("numberoffeature",len(pos))
print("numberofselectedfeature",sum(pos))
# Perform classification and store performance in P
model = RandomForestClassifier(n_estimators = 50, random_state = 42)
model.fit(X_selected_features_train, y_train_flatten)
n3 = dt.datetime.now()
y_pred_end=model.predict(X_selected_features_test)
n4 = dt.datetime.now()
# Compute performance

print("y_pred_end",y_pred_end)
print("y_test",y_test_flatten)
performance(y_test_flatten,y_pred_end)

print("feature selection time           : ",n2-n1)
print("training time with final features: ",n3-n2)
print("test time with final features    : ",n4-n3)

plot_cost_history(optimizer.cost_history)
plt.show()

