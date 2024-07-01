# Thesis Differentiable Inductive Logic Programming (∂ILP) for Fraud Detection

**Wolfson Boris, 2024**

# CONTENT:

## Data Sets
The PaySim data set is based on the work of 
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
├── DATA
│   └── paysim1  - PaySim dataset location
├── DILP
│   ├── examples - even number, and less than examples
│   └── src      - DILP main code
├── EDA          - EDA results and jupyter notebook
├── examples     - dilp datasets for dummy data, PaySim, and Recursion
├── report       - Poster and Thesis report
├── Snellius     - Snellius scripts to run examples
├── utils        - utility folder for dilp datasets generation
├── create_chain_fraud_no_merchants.ipynb
├── create_DT_and_DSC_datasets.ipynb Decision Tree and Deep Symbolic Classification 
├── create_tests.ipynb   - dummy set creation
├── fraud.parquet        - file for storing PaySim dataset
├── G19.ipynb            - G19 example from Evans Paper
├── kaggle.json          - Connection to kaggle
├── LICENSE
├── loading_dill_from_the_simulation.ipynb - loads model after running simulation
├── README.md
├── requirements.txt     - list of dependencies 
└── run_me_first.py      - downloads PaySim data set from Kaggle to DATA

```

