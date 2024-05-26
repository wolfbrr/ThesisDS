#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


get_ipython().run_cell_magic('time', '', 'term_x_0 = Term(True, \'X_0\')\nterm_x_1 = Term(True, \'X_1\')\ninput_dir = "../examples/chain-fraud/"\ninput_table = create_table(con, input_dir)\nprint(f\'ratio of positives %f\' % (100*input_table[\'Target\'].sum()/len(input_table)))\ntarget, p_e, constants, B, P, N = process_dir(input_dir)\n\np_a, rules = create_templates(p_e, target, term_x_0)\nrules[target] = (rules[target][0], Rule_Template(v=1, allow_intensional=True))\n\n\nlanguage_frame = Language_Frame(target, p_e, constants)\n# program_template = Program_Template(p_a, rules, T=10)\nprogram_template = Program_Template(p_a, rules, T=6)\n')


# In[3]:


print("DILP initialisation")
dilp = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
start_time = time.time()
dilp.train()
finish_time = time.time()
print("execution time %d" % (finish_time - start_time))



T=5
program_template = Program_Template(p_a, rules, T=T)
dilp = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
train(dilp)



