# Thesis Extracting Rules for Fraud Detection through Differential Inductive Logic Programming 

**Wolfson Boris, 2024**

# CONTENT:

## Data Sets
The data set is based on the work of 
E. A. Lopez-Rojas , A. Elmir, and S. Axelsson. "PaySim: A financial mobile money simulator for fraud detection". In: The 28th European Modeling and Simulation Symposium-EMSS, Larnaca, Cyprus. 2016
 
and can be located on the Kaggle site at:
https://www.kaggle.com/datasets/ealaxi/paysim1

by runining run\_me\_first.py data will be downloaded to DATA\paysim1 folder

## requirements
dill==0.3.8

duckdb==0.10.2

matplotlib==3.8.0

numpy==2.0.0

opendatasets==0.1.22

pyparsing==2.4.6

scikit\_learn==1.2.2

tensorflow==2.16.1

pandas

pyarrow

## directory and file structure

```
.
├── DATA  - PaySim dataset location
├── DILP  - DILP main code
├── EDA   - Explorative Data Analysis
├── examples - generated scenarions for testing
├── Snellius - Snellius scripts to run examples
├── utils    - utility folder for examples generation
├── README.md
├── requirements.txt - list of dependencies 
└── run_me_first.py  - downloads PaySim data set from Kaggle to DATA

```

