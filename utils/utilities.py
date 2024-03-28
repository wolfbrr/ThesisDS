"""
This module provides a function to convert tabular data to background knowledge, consisting of facts, positive and negative examples
"""
import os
import errno
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_facts_and_examples(df_, target, predicates, output_dir = "fraud-background"):
    """ create_facts_and_examples(df, target, predicates, output_dir = "fraud-background")

        Converts a tabular data to background knowledge: facts, positive, negative examp;es

        Parameters
        ----------
                    df : data frame
                target : name of target predicate, for positive/negative examples
            predicates : list of predicates for facts
            ouptut_dir : name of directory to save the data

        Returns
        -------
           saves three files to the provided directory: positive.dilp, negative.dilp, and facts.dilp

    """
    try:
        os.mkdir(output_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    df = df_.copy()   
    filter_ind = df[predicates].sum(axis=1)!=0
    df = df[filter_ind]
            
    tmp = df[df[target]==True].index.values
    print(target)
    print(tmp)
    np.savetxt(output_dir + '/positive.dilp', tmp, fmt=target+'(%d).') 

    tmp = df[df[target]==False].index.values
    np.savetxt(output_dir + '/negative.dilp', tmp, fmt=target+'(%d).')
    
    try:
        os.remove(output_dir + '/facts.dilp') #in the case there was already a file, it is needed to be removed
    except:
        pass
        
    with open(output_dir + '/facts.dilp', "a") as f:
        for q in predicates:
            print(q)
            tmp = df[df[q]==True].index.values
            print(tmp)
            np.savetxt(f, tmp, fmt= q +'(%d).')
            
    df_.to_parquet(path=output_dir + '/df.parquet')


def performance_metrics(y_pred, y_test, labels=[True, False]):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f'Accuracy: {accuracy:f}')
    print(f'Precision TP/(TP+FP): {precision:f}')
    print(f'Recall TP/(TP+FN): {recall:f}')
    print(f'F1 Score: {f1:f}')
    print(f'MCC Score: {mcc:f}')

    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot()
