#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../utils/") 
from utilities import performance_metrics
import time

from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP
import tensorflow as tf
import numpy as np
from src.utils import process_file, process_dir, create_table, output_rules, test_rule, train, test
from src.generate_template import create_templates
import duckdb
tf.random.set_seed(1000)
np.random.seed(1000)
import dill


# In[2]:


con = duckdb.connect(':memory:')
# enable automatic query parallelization
con.execute("PRAGMA threads=2")
# enable caching of parquet metadata
con.execute("PRAGMA enable_object_cache")


# Main idea: We need to learn a rule: 
# 
# Fraud_trans(U1, U2):- Transaction(X), Orig(X, U1), Dest(X, U2),  Fraud(X).
# 
# Fraud_chain(U1, U2):- Fraud_trans(U1, Y), Fraud_chain(Y, U2),

# 
# 

# In[4]:


term_x_0 = Term(True, 'X_0')
term_x_1 = Term(True, 'X_1')
input_dir = "../examples/chain-fraud/"
input_table = create_table(con, input_dir)
print(f'ratio of positives %f' % (100*input_table['Target'].sum()/len(input_table)))
target, p_e, constants, B, P, N = process_dir(input_dir)
p_e


# Fraud(Y) :- Fraud_chain(u1, u2), Orig(Y, u2) # v=2 (u1, u2)
# 
# Fraud_chain(u1, u2):- Fraud_trans(X, U2), pred_trans(U1, U2) #v=1 (X)
# 
# pred_trans(u1, u2) :-  Orig(Y, U1), Dest(Y, U2) #v=1 (Y)
# 
# Fraud_trans(X, U2):-  Dest(X, U2), Fraud(X). #v=1 (U2)

# In[5]:


p_a, rules = create_templates(p_e, target, term_x_0)
rules[target] = (rules[target][0], Rule_Template(v=2, allow_intensional=True))#Fraud(Y) :- Fraud_chain(u1, u2), Orig(Y, u2)

Fraud_trans = Atom([term_x_0, term_x_1], 'Fraud_trans')
Fraud_chain = Atom([term_x_0, term_x_1], 'Fraud_chain')
Pred_Transaction = Atom([term_x_0, term_x_1], 'Pred_Transaction')

p_a.append(Fraud_trans)
p_a.append(Fraud_chain)
p_a.append(Pred_Transaction)

rules[Fraud_chain] = (Rule_Template(v=1, allow_intensional=True), None) #Fraud_chain(u1, u2):- Fraud_trans(X, U2), pred_trans(U1, U2),pred_trans(u1, u2) :-  Orig(Y, U1), Dest(Y, U2) 
rules[Fraud_trans] = (Rule_Template(v=1, allow_intensional=True), None) #Fraud_trans(X, U2):-  Dest(X, U2), Fraud(X).
rules[Pred_Transaction] = (Rule_Template(v=1, allow_intensional=False), None) #Orig(Y, U1), Dest(Y, U2) #v=1 (Y)

language_frame = Language_Frame(target, p_e, constants)
# program_template = Program_Template(p_a, rules, T=10)
program_template = Program_Template(p_a, rules, T=6)


# In[6]:


rules


# In[6]:


print("DILP initialisation")
dilp = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
start_time = time.time()
dilp.train()
finish_time = time.time()
print("execution time %d" % (finish_time - start_time))


# In[39]:


abcd_rules = dilp.show_definition()
sql_str = output_rules(abcd_rules)
input_table = create_table(con, input_dir)
# predicted_table = test_rule(con, sql_str, target_predicate="Target")
# performance_metrics(predicted_table["Target"],input_table["Target"], labels=[True,False])


# In[ ]:




