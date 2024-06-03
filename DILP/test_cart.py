#!/usr/bin/env python
# coding: utf-8

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
input_dir = '../examples/fraud-cart-short/'

target, p_e, constants, B, P, N = process_dir(input_dir)
print(p_e)
print(target)


T=5
p_a, rules = create_templates(p_e[:6], target, term_x_0)

rules[target] = (rules[target][0], rules[target][0])

for predicate in p_a:
    rules[predicate] = (rules[predicate][0], rules[predicate][0])

language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template(p_a, rules, T=T)

print("DILP initialisation")
dilp_p6_T5 = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
print("DILP train")

start_time = time.time()
dilp_p6_T5.train()
finish_time = time.time()
print("execution time %d" % (finish_time - start_time))




derived_rules = dilp.show_definition()
sql_str = output_rules(derived_rules)


