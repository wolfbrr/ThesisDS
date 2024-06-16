#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../utils/") 
import time

from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP
import tensorflow as tf
import numpy as np
from src.utils import process_file, process_dir, create_table, output_rules, test_rule, train, test
from src.generate_template import create_templates
tf.random.set_seed(1000)
np.random.seed(1000)


term_x_0 = Term(True, 'X_0')
term_x_1 = Term(True, 'X_1')
input_dir = "../examples/simple-transitivity-fraud-rule/"
target, p_e, constants, B, P, N = process_dir(input_dir)
print(p_e)
print(target)

#--------------- Template Definition to change-------------------------------
T=6
p_a, rules = create_templates(p_e, target, term_x_0)
print(p_a)
rules[target] = (Rule_Template(v=0, allow_intensional=False), Rule_Template(v=1, allow_intensional=True))

language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template(p_a, rules, T=T)

#------------------------------------------------------
print(rules)
print(f'target[0].v=%d, allow_intentsional=%d' % (rules[target][0].v, rules[target][0].allow_intensional))
print("DILP initialisation")
dilp = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
print("DILP train")

start_time = time.time()
dilp.train()
finish_time = time.time()
print("execution time %d" % (finish_time - start_time))




derived_rules = dilp.show_definition()
sql_str = output_rules(derived_rules)


