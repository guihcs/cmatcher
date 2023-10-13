import ply.lex as lex

tokens = (
    'ID',
    'RESTRICTION',
    'INVERSE',
    'INTERSECTION',
    'UNION',
    'ONPROP',
    'SOMEVAL',
    'COMPLEMENT',
    'VALUE'
)

literals = ['=', '(', ')', ',']

reserved = {
    'Restriction': 'RESTRICTION',
    'InverseOf': 'INVERSE',
    'IntersectionOf': 'INTERSECTION',
    'UnionOf': 'UNION',
    'onProperty': 'ONPROP',
    'someValuesFrom': 'SOMEVAL',
    'ComplementOf': 'COMPLEMENT',
    'value': 'VALUE'
}

def t_ID(t):
    r'[-:\w]+'
    t.type = reserved.get(t.value, 'ID')
    return t

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t\n'


# Error handling rule
def t_error(t):
    raise Exception("Illegal character '%s'" % t.value[0])


# Build the lexer
lexer = lex.lex()


# # Test it out
# data = '''
# IntersectionOf(
#     Written_contribution,
#     ComplemetOf(
#         Abstract
#     )
# )
# '''
#
# # Give the lexer some input
# lexer.input(data)
#
# # Tokenize
#
# while True:
#     tok = lexer.token()
#     if not tok:
#         break  # No more input
#     print(tok)