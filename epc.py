# Yacc example

import ply.yacc as yacc

from elex import tokens
# from elex import tokens


def p_s(p):
    '''s : classexp
    '''
    p[0] = p[1]


def p_classexp(p):
    '''classexp : class
                | intersection
                | union
                | restriction
                | complement'''
    p[0] = p[1]

def p_class(p):
    'class : ID'
    p[0] = ('class', p[1])


def p_intersection(p):
    'intersection : INTERSECTION "(" classlist ")"'
    p[0] = ('intersection', p[3])

def p_union(p):
    'union : UNION "(" classlist ")"'
    p[0] = ('union', p[3])

def p_restriction(p):
    '''restriction : RESTRICTION "(" ONPROP "=" propexp "," SOMEVAL "=" classexp ")"
                   | RESTRICTION "(" ONPROP "=" propexp "," VALUE "=" classexp ")"
   '''

    p[0] = ('restriction', p[5], p[7], p[9])

def p_complement(p):
    'complement : COMPLEMENT "(" classexp ")"'
    p[0] = ('complement', p[3])

def p_propexp(p):
    '''propexp : ID
               | INVERSE "(" ID ")"'''
    mod = None
    if p[1] == 'InverseOf':
        mod = 'inverse'
        prop = p[3]
    else:
        prop = p[1]

    p[0] = ('prop', mod, prop)


def p_classlist(p):
    '''classlist : classexp "," classlist
                 | classexp '''
    if len(p) < 4:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


# Error rule for syntax errors
def p_error(p):
    print(p, parser.symstack[-5:])
    raise Exception("Syntax error in input!")


# Build the parser
parser = yacc.yacc()
