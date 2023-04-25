# PSO with NNs (Combine different method with selected features using PSO)

import pandas as pd
import datetime as dt
import numpy as np
import pyswarms as ps
from pyswarm import pso
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Concatenate, Dropout
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from keras.optimizers import SGD
from keras.models import Model

name= "Ensemble ML using PSO"
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

n1 = dt.datetime.now()

##### MODEL 1 ##### MLP #####
def MLP ():
    # SELECTED VARIABLES #
    pos = [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    pos=np.array(pos)
    X_train_mlp = X_train[:,pos==1]  # subset
    X_test_mlp = X_test[:,pos==1]  # subset
    # SELECTED VARIABLES #
    input = Input(shape=(X_train_mlp.shape[1],))
    hidden1 = Dense(X_train_mlp.shape[1] + 2, activation="sigmoid")(input)
    hidden2 = Dense(X_train_mlp.shape[1] + 1, activation="sigmoid")(hidden1)
    hidden3 = Dense(round(float(X_train_mlp.shape[1]) / 2), activation="sigmoid")(hidden2)
    output = Dense(y_train.shape[1], activation="sigmoid")(hidden3)
    model1 = Model(inputs=[input], outputs=[output])
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9)
    model1.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    history = model1.fit(X_train_mlp, y_train, epochs=100)
    score = model1.evaluate(X_test_mlp, y_test)
    y_train_pred1=model1.predict(X_train_mlp)
    y_test_pred1=model1.predict(X_test_mlp)
    y_train_pred1_class=np.argmax(y_train_pred1, axis=-1)
    y_test_pred1_class=np.argmax(y_test_pred1, axis=-1)
    return y_train_pred1, y_test_pred1,y_train_pred1_class,y_test_pred1_class
##### MODEL 1 ##### MLP #####

##### MODEL 2 ##### CNN #####
def CNN ():
    # SELECTED VARIABLES #
    pos=[1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0]
    pos=np.array(pos)
    X_train_cnn = X_train[:,pos==1]  # subset
    X_test_cnn = X_test[:,pos==1]  # subset
    # SELECTED VARIABLES #
    input = Input(shape=(X_train_cnn.shape[1], 1))
    conv1 = Conv1D(20, round(X_train_cnn.shape[1] / 2), activation="sigmoid")(input)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv1D(4, round(X_train_cnn.shape[1] / 2) - 2, activation="sigmoid")(batch1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D()(batch2)
    flat = Flatten()(pool2)
    output = Dense(y_train.shape[1], activation='sigmoid')(flat)
    model2 = Model(inputs=[input], outputs=[output])
    model2.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    model2.summary()
    history = model2.fit(X_train_cnn, y_train, epochs=20)
    score = model2.evaluate(X_test_cnn, y_test)
    y_train_pred2 = model2.predict(X_train_cnn)
    y_test_pred2 = model2.predict(X_test_cnn)
    y_train_pred2_class=np.argmax(y_train_pred2, axis=-1)
    y_test_pred2_class=np.argmax(y_test_pred2, axis=-1)
    return y_train_pred2, y_test_pred2,y_train_pred2_class,y_test_pred2_class
##### MODEL 2 ##### CNN #####

##### MODEL 3 ##### RF #####
def RF ():
    # SELECTED VARIABLES #
    pos=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1]
    pos = np.array(pos)
    X_train_rf = X_train[:, pos == 1]  # subset
    X_test_rf = X_test[:, pos == 1]  # subset
    # SELECTED VARIABLES #
    model3 = RandomForestClassifier(n_estimators = 50, random_state = 42)
    #n_estimators= The number of trees in the forest.
    #random_state= Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
    model3.fit(X_train_rf, np.ravel(y_train_flatten))
    y_train_pred3 = model3.predict_proba(X_train_rf)  # return class probabilities
    y_test_pred3 = model3.predict_proba(X_test_rf)  # return class probabilities
    y_train_pred3_class = model3.predict(X_train_rf)  # return the class label like 0 or 1
    y_test_pred3_class = model3.predict(X_test_rf)  # return the class label like 0 or 1
    return y_train_pred3, y_test_pred3, y_train_pred3_class, y_test_pred3_class
##### MODEL 3 ##### RF #####

##### MODEL 4 ##### SVM #####
def SVM ():
    # SELECTED VARIABLES #
    pos=[1,0,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1]
    pos = np.array(pos)
    X_train_svm = X_train[:, pos == 1]  # subset
    X_test_svm = X_test[:, pos == 1]  # subset
    # SELECTED VARIABLES #
    model4 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='scale', max_iter=- 1)
    model4.fit(X_train_svm, y_train_flatten)
    y_train_pred4 = model4.predict_proba(X_train_svm)
    y_test_pred4 = model4.predict_proba(X_test_svm)
    y_train_pred4_class = model4.predict(X_train_svm)  # return the class label like 0 or 1
    y_test_pred4_class = model4.predict(X_test_svm)  # return the class label like 0 or 1
    return y_train_pred4, y_test_pred4, y_train_pred4_class, y_test_pred4_class
##### MODEL 4 ##### SVM #####

##### MODEL 5 ##### GB #####
def GB ():
    # SELECTED VARIABLES #
    pos=[0,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1]
    pos = np.array(pos)
    X_train_gb = X_train[:, pos == 1]  # subset
    X_test_gb = X_test[:, pos == 1]  # subset
    # SELECTED VARIABLES #
    model5 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
    #n_estimators= The number of trees in the forest.
    #random_state= Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
    model5.fit(X_train_gb, np.ravel(y_train_flatten))
    y_train_pred5 = model5.predict_proba(X_train_gb)  # return class probabilities
    y_test_pred5 = model5.predict_proba(X_test_gb)  # return class probabilities
    y_train_pred5_class = model5.predict(X_train_gb)  # return the class label like 0 or 1
    y_test_pred5_class = model5.predict(X_test_gb)  # return the class label like 0 or 1
    return y_train_pred5, y_test_pred5, y_train_pred5_class, y_test_pred5_class
##### MODEL 5 ##### GB #####

jtut=[]

# Define the objective (to be minimize)
def f_per_particle(x):
    sumx=np.sum(x)
    y = [round(x[0]/sumx, 2), round(x[1]/sumx, 2), round(x[2]/sumx, 2), round(x[3]/sumx, 2), round(x[4]/sumx, 2)]
    y_train_preds_last_zero=np.round(y_train_preds_zero.dot(y), 4)
    y_train_preds_last_one=np.round(y_train_preds_one.dot(y), 4)
    y_train_preds_last = np.array([np.array(y_train_preds_last_zero).T,np.array(y_train_preds_last_one).T]).T
    y_train_preds_last=np.argmax(y_train_preds_last, axis=-1)
    train_performance = round(0.6*(accuracy_score(y_train_flatten, y_train_preds_last))+0.4*roc_auc_score(y_train_flatten, y_train_preds_last, multi_class='ovr', average="macro"),5)
    j = 1 - train_performance
    jtut.append(j)
    return j

def f(x):
    j = [f_per_particle(x[i]) for i in range(swarmsize)]
    return np.array(j)

models = [MLP(), CNN(), RF(), SVM(), GB()]
y_train_preds_zero=[]
y_train_preds_one=[]
y_train_preds_class=[]
y_test_preds_zero=[]
y_test_preds_one=[]
y_test_preds_class=[]
accuracylist_train=[]
ROClist_train=[]
accuracylist_test=[]
ROClist_test=[]

for x in range(0, len(models)):
    y_train_preds_zero.append(np.resize(models[x][0][:,0], (1, len(X_train))))
    y_train_preds_one.append(np.resize(models[x][0][:,1],(1,len(X_train))))

    y_test_preds_zero.append(np.resize(models[x][1][:,0], (1, len(X_test))))
    y_test_preds_one.append(np.resize(models[x][1][:,1],(1,len(X_test))))

    y_train_preds_class.append(np.resize(models[x][2],(1,len(X_train))))
    y_test_preds_class.append(np.resize(models[x][3],(1,len(X_test))))

    accuracylist_train.append(round(accuracy_score(y_train_flatten, y_train_preds_class[x].T), 3))
    ROClist_train.append(round(roc_auc_score(y_train_flatten,y_train_preds_class[x].T , multi_class='ovr', average="macro"), 3))
    accuracylist_test.append(round(accuracy_score(y_test_flatten, y_test_preds_class[x].T), 3))
    ROClist_test.append(round(roc_auc_score(y_test_flatten,y_test_preds_class[x].T , multi_class='ovr', average="macro"), 3))
print("accuracylist_train", accuracylist_train)
print("ROClist_train", ROClist_train)
print("accuracylist_test", accuracylist_test)
print("ROClist_test", ROClist_test)
y_train_preds_zero = np.concatenate(y_train_preds_zero, axis=0).T
y_train_preds_one = np.concatenate(y_train_preds_one, axis=0).T
y_test_preds_zero = np.concatenate(y_test_preds_zero, axis=0).T
y_test_preds_one = np.concatenate(y_test_preds_one, axis=0).T
n2 = dt.datetime.now()

dimensions = 5
swarmsize= 5
maxiter= 100


min_bound = np.zeros(dimensions)
max_bound = np.ones(dimensions)

bounds = (min_bound, max_bound)

options = {'c1': 0.5, 'c2': 0.3, 'w':0.5, 'k': 3, 'p':1}
optimizer = ps.single.GlobalBestPSO(n_particles=swarmsize, dimensions=dimensions, options=options, bounds=bounds)
fopt, xopt = optimizer.optimize(f, iters=maxiter)


sumx = np.sum(xopt)
weights = [round(xopt[0] / sumx, 2), round(xopt[1] / sumx, 2), round(xopt[2] / sumx, 2), round(xopt[3] / sumx, 2), round(xopt[4] / sumx, 2)]
print("fopt", fopt)
print("xopt",xopt)
print("weights", weights)

n3 = dt.datetime.now()


# Perform classification and store performance in P
y_train_preds_end_zero = np.round(y_train_preds_zero.dot(weights), 4)
y_train_preds_end_one = np.round(y_train_preds_one.dot(weights), 4)
y_train_pred_end = np.array([np.array(y_train_preds_end_zero).T, np.array(y_train_preds_end_one).T]).T
y_train_pred_end = np.argmax(y_train_pred_end, axis=-1)

subset_performance = (y_train_pred_end == y_train_flatten).mean()
print('Train Subset performance: %.3f' % (subset_performance))

# Compute performance
y_test_preds_end_zero = np.round(y_test_preds_zero.dot(weights), 4)
y_test_preds_end_one = np.round(y_test_preds_one.dot(weights), 4)
y_test_pred_end = np.array([np.array(y_test_preds_end_zero).T, np.array(y_test_preds_end_one).T]).T
y_test_pred_end = np.argmax(y_test_pred_end, axis=-1)

subset_performance = (y_test_pred_end == y_test_flatten).mean()
print('Test Subset performance: %.3f' % (subset_performance))
n4 = dt.datetime.now()

##### Performance Metrics #####
from Performance_Ensemble import performance_metrics
# y_test is real class values and should be in categorical form
# y_test_pred is predicted values and consists of "#ofrun" prediction arrays containing single class values.
y_test_pred_end=[np.array(y_test_pred_end)]
performance_metrics(name,numberofrun,numberofclass,y_test,y_test_pred_end)

print("training time of models                   : ",n2-n1)
print("optimization time for ensemble structure  : ",n3-n2)
print("test time with final weights              : ",n4-n3)

costtut=[1]

for i in range(0,maxiter):
    costiter=min(jtut[i * swarmsize:(i + 1) * swarmsize])
    if costiter<=min(costtut):
        costtut.append(costiter)
    else:
        costtut.append(min(costtut))

x = np.arange(0,len(costtut)-1)
y= costtut[1:len(costtut)]
plt.plot(x, y,color='black', marker='*')
plt.xlabel('Iteration Number')
plt.ylabel('Loss Function')
plt.show()

plot_cost_history(optimizer.cost_history)
plt.show()
