import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


def performance_metrics(name, run, numberofclass, outputMatrix, Test_Yd_calculated_Class):
    print(" ")
    print("Performance criteria - {0}"''.format(name))
    print(" ")

    if numberofclass==2:
        class_status="binary"
    else:
        class_status="macro"

    if run == 1:
        # The number of rows of y_test and y_pred must be equal to the number of samples, the number of columns must be equal to the number of columns of the outputset
        outputMatrix = np.array(outputMatrix).reshape(len(outputMatrix), len(outputMatrix[0]))
        if Test_Yd_calculated_Class[0].ndim != 1:
            Test_Yd_calculated_Class = np.array(Test_Yd_calculated_Class).reshape(len(outputMatrix), len(outputMatrix[0]))
            Test_Yd_calculated_Class = np.argmax(Test_Yd_calculated_Class, axis=-1)  # convert the data to 0-1
        else:
            Test_Yd_calculated_Class=np.array(Test_Yd_calculated_Class[0])
        if len(outputMatrix[0]) != 1:
            outputMatrix = np.argmax(outputMatrix, axis=-1)  # convert the data to 0-1

        print("Real :", outputMatrix)
        print("Pred.:", Test_Yd_calculated_Class)

        # Confusion Matrix - verify accuracy of each class
        cm = confusion_matrix(outputMatrix, Test_Yd_calculated_Class)
        sns.set(font_scale=1.4)  # Adjust to fit
        sns.heatmap(cm,annot=True,fmt=".0f",linewidths=.5)
        label_font = {'size': '14'}  # Adjust to fit
        plt.xlabel('Predicted', fontdict=label_font)
        plt.ylabel('Actual', fontdict=label_font)
        plt.title('Confusion Matrix for {0}'''.format(name))
        accuracy = round(accuracy_score(outputMatrix, Test_Yd_calculated_Class), 3)
        recall = round(recall_score(outputMatrix, Test_Yd_calculated_Class, average=class_status), 3)
        precision = round(precision_score(outputMatrix, Test_Yd_calculated_Class, average=class_status), 3)
        fmeasure = round(f1_score(outputMatrix, Test_Yd_calculated_Class, average=class_status), 3)
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

        # One hot encode y values for neural network.
        from keras.utils import to_categorical
        outputMatrix = to_categorical(outputMatrix, num_classes=numberofclass, dtype ="int32")
        Test_Yd_calculated_Class = to_categorical(Test_Yd_calculated_Class, num_classes=numberofclass, dtype ="int32")
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

        colors = cycle(['pink', 'violet', 'lightgreen', 'yellow', 'aqua', 'darkorange', 'cornflowerblue', 'teal', 'lime', 'lightcoral'])
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
    else:
        cm_list=[]
        accuracy_list=[]
        recall_list=[]
        precision_list=[]
        fmeasure_list=[]
        roc_list=[]


        # The number of rows of y_test and y_pred must be equal to the number of samples, the number of columns must be equal to the number of columns of the outputset
        outputMatrix = np.array(outputMatrix).reshape(len(outputMatrix), len(outputMatrix[0]))
        for j in range(0, run):
            if Test_Yd_calculated_Class[j].ndim != 1:
                Test_Yd_calculated_Class[j] = np.array(Test_Yd_calculated_Class[j]).reshape(len(outputMatrix),len(outputMatrix[0]))
                Test_Yd_calculated_Class[j] = np.argmax(Test_Yd_calculated_Class[j], axis=-1)  # convert the data to 0-1
            else:
                Test_Yd_calculated_Class[j] = np.array(Test_Yd_calculated_Class[j])
        if len(outputMatrix[0]) != 1:
            outputMatrix = np.argmax(outputMatrix, axis=-1)  # convert the data to 0-1

        print("Real :", outputMatrix)
        print("Pred.:", Test_Yd_calculated_Class)
        labels = list(range(0, max(outputMatrix) + 1))
        outputMatrix_b = label_binarize(outputMatrix, classes=labels)

        for j in range(0, run):
            # Confusion Matrix - verify accuracy of each class
            cm = confusion_matrix(outputMatrix, Test_Yd_calculated_Class[j])
            cm_list.append(cm)
            accuracy = round(accuracy_score(outputMatrix, Test_Yd_calculated_Class[j]), 3)
            recall = round(recall_score(outputMatrix, Test_Yd_calculated_Class[j], average=class_status), 3)
            precision = round(precision_score(outputMatrix, Test_Yd_calculated_Class[j], average=class_status), 3)
            fmeasure = round(f1_score(outputMatrix, Test_Yd_calculated_Class[j], average=class_status), 3)
            Test_Yd_calculated_Class_b = label_binarize(Test_Yd_calculated_Class[j], classes=labels)
            roc = round(roc_auc_score(outputMatrix_b, Test_Yd_calculated_Class_b, multi_class='ovr', average="macro"),3)
            accuracy_list.append(accuracy)
            recall_list.append(recall)
            precision_list.append(precision)
            fmeasure_list.append(fmeasure)
            roc_list.append(roc)
            my_dict = {"Accuracy": accuracy,
                       "Recall": recall,
                       "Precision": precision,
                       "F-measure": fmeasure,
                       "Roc": roc}
            print("Run number {0}:".format(j+1))
            print(my_dict)
        print("")
        print("Average:")
        my_dict1 = {"Accuracy_Avg": round(np.mean(accuracy_list),3),
                   "Recall_Avg": round(np.mean(recall_list),3),
                   "Precision_Avg": round(np.mean(precision_list),3),
                   "F-measure_Avg": round(np.mean(fmeasure_list),3),
                    "Roc_Avg": round(np.mean(roc_list),3)}
        print(my_dict1)
        print("")
        print("Std:")
        my_dict2 = {"Accuracy_Std": round(np.std(accuracy_list), 3),
                   "Recall_Std": round(np.std(recall_list), 3),
                   "Precision_Std": round(np.std(precision_list), 3),
                   "F-measure_Std": round(np.std(fmeasure_list), 3),
                    "Roc_Std": round(np.std(roc_list), 3)}
        print(my_dict2)
        print("")
        print("Confusion Matrix:")
        print("Confusion Matrix for each run:")
        print(cm_list)
        cm=0
        for k in range(0,run):
            cm=cm+cm_list[k]
        cm=(cm/run)
        cm=np.round(cm,2)
        print("Confusion Matrix Avg:")
        print(cm)
        sns.set(font_scale=1.4)  # Adjust to fit
        sns.heatmap(cm,annot=True,fmt=".1f",linewidths=.5)
        label_font = {'size': '14'}  # Adjust to fit
        plt.xlabel('Predicted', fontdict=label_font)
        plt.ylabel('Actual', fontdict=label_font)
        plt.title('Confusion Matrix (Avg.) for {0}'''.format(name))
        tpr_calc = []
        fpr_calc = []
        TP = []
        FP = []
        TN = []
        FN = []
        for p in range(numberofclass):
            # Calculate True Positive
            TP.append(cm[p][p])
            FP.append(np.sum(cm, axis=0)[p] - TP[p])
            FN.append(np.sum(cm, axis=1)[p] - TP[p])
            TN.append(np.sum(cm) - TP[p] - FP[p] - FN[p])
        for p in range(numberofclass):
            # Calculate True Positive Rate
            tpr_alt = TP[p] + FN[p]
            if tpr_alt == 0:
                tpr_alt = 1
            tpr_calc.append(TP[p] / tpr_alt)
            # Calculate False Positive Rate
            fpr_alt = FP[p] + TN[p]
            if fpr_alt == 0:
                fpr_alt = 1
            fpr_calc.append(FP[p] / fpr_alt)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        FPR_avg = []
        TPR_avg = []

        for p in range(0, numberofclass):
            FPR_avg.append(np.array([0, fpr_calc[p], 1]))
            TPR_avg.append(np.array([0, tpr_calc[p], 1]))

        for p in range(0, numberofclass):
            fpr[p] = FPR_avg[p]
            tpr[p] = TPR_avg[p]

        for i in range(0, numberofclass):
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numberofclass)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(numberofclass):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= numberofclass

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw = 2
        plt.figure(figsize=(8, 5))
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='green', linestyle=':', linewidth=4)

        colors = cycle(['pink', 'violet', 'lightgreen', 'yellow', 'aqua', 'darkorange', 'cornflowerblue', 'teal', 'lime', 'lightcoral'])
        for i, color in zip(range(numberofclass), colors):
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
    return my_dict