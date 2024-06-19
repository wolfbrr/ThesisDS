"""
This module provides a function to convert tabular data to background knowledge, consisting of facts, positive and negative examples
"""
import os
import errno
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def output_predicate(q, df, f, flag=True):
    """ Formated output of a predicate from the data frame"""
    # variables = q.split('--')
    variables = re.split('--|->', q)
    tmp = df[df[q]==flag]
    if len(variables)==2:
        for i,j in zip(tmp.index.values, tmp[variables[1]]):
            print(q, i, j, 'flag=', flag)
            f.writelines(f'%s(%d,%d).\n' % (q,i,j))
    elif len(variables)>2:
        for i,j in zip(tmp[variables[1]], tmp[variables[2]]):
            print(q, i, j, 'flag=', flag)
            f.writelines(f'%s(%d,%d).\n' % (q,i,j))
    else:
        print(tmp.index.values)
        np.savetxt(f, tmp.index.values, fmt= q +'(%d).')

def create_facts_and_examples(df_, target, predicates, output_dir = "fraud-background", filter_null_columns=True):
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
    try:
        os.remove(output_dir + '/facts.dilp') #in the case there was already a file, it is needed to be removed
    except:
        pass
    
    try:
        os.remove(output_dir + '/positive.dilp') #in the case there was already a file, it is needed to be removed
    except:
        pass
    
    try:
        os.remove(output_dir + '/negative.dilp') #in the case there was already a file, it is needed to be removed
    except:
        pass
        
    df = df_.copy()

    # filter out the lines with facts that are False, fact file includes True examples for predicates, 
    # for 1 arity some there is a chance that the constants won't appear in the facts but will in Positive,negative examples
    # for recursion, that might not be the case
    if filter_null_columns:
            
        filter_ind = df[predicates].sum(axis=1)!=0
    
        df = df[filter_ind]

    tmp = df[df[target]==True].index.values

    with open(output_dir + '/positive.dilp', "a") as f:
        output_predicate(q=target, df=df, f=f, flag=True)

    with open(output_dir + '/negative.dilp', "a") as f:
        output_predicate(q=target, df=df, f=f, flag=False)

    with open(output_dir + '/facts.dilp', "a") as f:
        for q in predicates:
            print(q)
            output_predicate(q=q, df=df, f=f, flag=True)


    df_.to_parquet(path=output_dir + '/df.parquet')

def performance_metrics(y_pred, y_test, labels=[True, False], title=''):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f'Performance of %s' % title)
    print(f'Accuracy: {accuracy:f}')
    print(f'Precision TP/(TP+FP): {precision:f}')
    print(f'Recall TP/(TP+FN): {recall:f}')
    print(f'F1 Score: {f1:f}')
    print(f'MCC Score: {mcc:f}')

    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,  display_labels=['Positive','Negative'])
    disp.plot()
    plt.title(title)
