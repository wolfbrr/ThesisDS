from src.ilp.template import Rule_Template
from src.core import Term, Atom

def create_templates(p_e, target, term_x_0):
    p_a=[]
    rules={}

    for i in range(1,len(p_e)-1):
        name = 'pred'+str(i)
        p_a.append(Atom([term_x_0], name))

    for p_a_ in p_a:
        rules[p_a_]=(Rule_Template(v=0, allow_intensional=True), None)

    if len(p_a)>0:
        rules[p_a_]=(Rule_Template(v=0, allow_intensional=False), None) #last rule should not include any intensional predicate
        target_rule = (Rule_Template(v=0, allow_intensional=True),None)

    else:
        target_rule = (Rule_Template(v=0, allow_intensional=False),None)


    rules[target]=target_rule #last rule should not include any intensional predicate
    return p_a, rules
