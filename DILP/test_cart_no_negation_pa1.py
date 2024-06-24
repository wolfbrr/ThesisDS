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
import duckdb
import dill
con = duckdb.connect(':memory:')
# enable automatic query parallelization
con.execute("PRAGMA threads=2")
# enable caching of parquet metadata
con.execute("PRAGMA enable_object_cache")

tf.random.set_seed(1000)
np.random.seed(1000)


term_x_0 = Term(True, 'X_0')
input_dir = '../examples/fraud-cart-short-no-negation/'


target, p_e, constants, B, P, N = process_dir(input_dir)
print(p_e)
print(target)


T=5
p_a, rules = create_templates(p_e[:-2], target, term_x_0)

language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template(p_a, rules, T=T)

print("DILP initialisation")
dilp = DILP(language_frame, B, P, N, program_template, allow_target_recursion=True)
print("DILP train")
train(dilp)

print("train performance")
test(dilp, create_table(con, input_dir, 'df'), con)
print("test performance")
test(dilp, create_table(con, input_dir, 'df_test'), con)





