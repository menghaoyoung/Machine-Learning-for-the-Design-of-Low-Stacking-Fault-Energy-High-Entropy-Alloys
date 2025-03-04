# -*- coding: utf-8 -*-
"""

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve_multiclass(y_true, y_score):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))

    for i in range(len(lb.classes_)):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve (area = %0.2f) for class %s' % (roc_auc[i], lb.classes_[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Load training and testing sets
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    
    X_train, y_train = train_data.iloc[:, :13], train_data.iloc[:, 13]
    X_test, y_test = test_data.iloc[:, :13], test_data.iloc[:, 13]

    # Standardized numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select classifier
    classifier = GradientBoostingClassifier(n_estimators=475, learning_rate=0.03807947176588889, max_features="sqrt" ,max_depth=3, random_state=42)

    # Train classifier
    classifier.fit(X_train, y_train)

    # prediction
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy：", accuracy)

    # Output confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix：")
    print(cm)

    # Draw a confusion matrix diagram
    plt.figure()
    plot_confusion_matrix(cm, classes=['0', '1'], title='Confusion matrix')
    plt.show()

    # Calculate ROC curve parameters
    y_score = classifier.predict_proba(X_test)

    # Draw multi class ROC curves
    plot_roc_curve_multiclass(y_test, y_score)

if __name__ == "__main__":
    main()


