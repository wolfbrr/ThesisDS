import os
import errno
import opendatasets as od

# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/datasets/ealaxi/paysim1'# Using opendatasets let's download the data sets
od.download(dataset)
try:
    os.mkdir('DATA/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise

os.system('rm -rf DATA/paysim1')
os.system('mv paysim1/ DATA/paysim1')