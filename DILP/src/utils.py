'''Defines stateless utility functions
'''
import sys

sys.path.append("../utils/")

from src.core import Atom
import pyparsing as pp
from src.core import Term, Atom
import os, time
from utilities import performance_metrics

def is_intensional(atom: Atom):
    '''Checks if the atom is intensional. If true returns true, otherwise returns false

    Arguments:
        atom {Atom} -- Atom to be analyzed
    '''
    for term in atom.terms:
        if not term.isVariable:
            return False

    return True


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    total -= 1
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


INTENSIONAL_REQUIRED_MESSAGE = 'Atom is not intensional'

def process_file(filename):
    # relationship will refer to 'track' in all of your examples
    relationship = pp.Word(pp.nums + '-' + ' ' + '.' + pp.alphas + '_' +'{' + '}' + ',').setResultsName('relationship', listAllMatches=True)

    number = pp.Word(pp.nums + '.')
    variable = pp.Word(pp.alphas)
    # an argument to a relationship can be either a number or a variable
    argument = number | variable

    # arguments are a delimited list of 'argument' surrounded by parenthesis
    arguments = (pp.Suppress('(') + pp.delimitedList(argument) +
                 pp.Suppress(')')).setResultsName('arguments', listAllMatches=True)

    # a fact is composed of a relationship and it's arguments
    # (I'm aware it's actually more complicated than this
    # it's just a simplifying assumption)
    fact = (relationship + arguments).setResultsName('facts', listAllMatches=True)

    # a sentence is a fact plus a period
    sentence = fact + pp.Suppress('.')

    # self explanatory
    prolog_sentences = pp.OneOrMore(sentence)

    atoms = []
    predicates = set()
    constants = set()
    with open(filename) as f:
        data = f.read().replace('\n', '')
        result = prolog_sentences.parseString(data)
        print(len(result['facts']))
        
        for idx in range(len(result['facts'])):
            fact = result['facts'][idx]
            predicate = result['relationship'][idx]
            terms = [Term(False, term) for term in result['arguments'][idx]]
            term_var = [Term(True, f'X_{i}') for i in range(len(terms))]

            predicates.add(Atom(term_var, predicate))
            atoms.append(Atom(terms, predicate))
            constants.update([term for term in result['arguments'][idx]])
    return atoms, predicates, constants

def process_dir(input_dir):
    B, pred_f, constants_f = process_file('%s/facts.dilp' % input_dir)
    print("end of facts processing")

    P, target_p, constants_p = process_file(
        '%s/positive.dilp' % input_dir)
    print("end of positive examples processing")

    N, target_n, constants_n = process_file(
        '%s/negative.dilp' % input_dir)
    print("end negative examples processing")

    if not (target_p == target_n):
        raise Exception('Positive and Negative files have different targets')
    elif not len(target_p) == 1:
        raise Exception('Can learn only one predicate at a time')
    elif not constants_n.issubset(constants_f) or not constants_p.issubset(constants_f):
        raise Exception(
            'Constants not in fact file exists in positive/negative file')

    print("data is in order")
    target = list(target_p)[0]
    p_e = list(pred_f)
    constants = constants_f
    return target, p_e, constants, B, P, N


def output_rules(rules):
    """ Print induced rules and convert them to a sql query"""
    sql_query="select"
    for rule in rules[::-1]:
        for i in rule:
            if i==None:
                pass
            else:
                print(i.head.predicate,":",i.body[0].predicate,i.body[1].predicate)
                sql_query+=f" \"%s\" and \"%s\" as \"%s\",\n" % \
                                    (i.body[0].predicate, i.body[1].predicate, i.head.predicate)
    print(sql_query)
    return sql_query


def create_table(con, input_dir, name=''):
    """ read a parquet file and return an SQL table my_table """
    try:
        con.sql('DROP TABLE my_table')
        print("previous table dropped")
    except:
        pass
    if name=='':
        con.sql(f"CREATE TABLE my_table AS SELECT * FROM '%s'" % os.path.join(input_dir,'df.parquet'))
    else:
        con.sql(f"CREATE TABLE my_table AS SELECT * FROM '%s'" % os.path.join(input_dir,f'%s.parquet' % name))

    return con.sql("""select * from my_table""").df()

def test_rule(con, sql_str, target_predicate="Target"):
    """Test rule on the pandas data frame"""

    df=con.sql(sql_str + 'from my_table ').df()
    return df[[target_predicate]]

def train(dilp):
    """ training encapsulation for DILP"""
    start_time = time.time()
    dilp.train()
    finish_time = time.time()
    print("execution time %d" % (finish_time - start_time))

def test(dilp, input_table, con):
    """ testing encapsulation for DILP versus input table"""

    rules = dilp.show_definition()
    sql_str = output_rules(rules)
    predicted_table = test_rule(con, sql_str, target_predicate=dilp.language_frame.target.predicate)
    performance_metrics(predicted_table[dilp.language_frame.target.predicate], input_table[dilp.language_frame.target.predicate], labels=[True,False])
    return sql_str
