from io import StringIO
from contextlib import redirect_stdout


top_code_str = '''
result = special_print()
print("top level agent", result)
'''

inner_code_str = '''
import numpy as np

def euclidean_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

red_block_position = [2.0, 3.0, 4.0]
filtered_bowls = [{'position': [1.0, 2.0, 3.0]}, {'position': [2.0, 3.0, 4.0]}]

distances = [euclidean_distance(red_block_position, bowl['position']) for bowl in filtered_bowls]

result = normal_print()
print("inner agent", result)
'''

def normal_print():
    string = "Hello World normal"
    print(string)
    return string

def special_print():
    string = "Hello World special"
    print(string)

    custom_gvars = {normal_print.__name__: normal_print}
    lvars = None

    out = StringIO()
    with redirect_stdout(out):
        exec_after_defining(inner_code_str, custom_gvars, lvars)

    return out.getvalue()

import ast

def split_code(code):
    tree = ast.parse(code)
    definitions = []
    rest = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions.append(node)
        else:
            rest.append(node)

    def unparse(nodes):
        return "\n".join([ast.unparse(node) for node in nodes])

    definitions_code = unparse(definitions)
    rest_code = unparse(rest)

    return definitions_code, rest_code

def exec_after_defining(code_string, global_vars, local_vars):
    definitions_code, rest_code = split_code(code_string)
    exec(definitions_code, global_vars, local_vars)
    global_vars.update(local_vars)
    exec(rest_code, global_vars, local_vars)


first_code = '''
c = 3
'''

# The arbitrary code string
code_string = '''
def print_hello():
    print("Hello")
    return "Hello"
b = c
# print(locals())
print_hello()
a = [print_hello() for i in range(b)]
d = a[0]
e = d
'''


print("new mthod")
global_vars = {}
local_vars = None

exec(first_code, global_vars, local_vars)
print(global_vars['c'])
exec(code_string, global_vars, local_vars)
print(global_vars['e'])
print("---------------------")






# Split the code into definitions and the rest
definitions_code, rest_code = split_code(code_string)
print('definitions', definitions_code)
print('rest', rest_code)


# Define the execution context dictionaries
local_vars = {'c': 3}
global_vars = {}

# exec(first_code, global_vars, local_vars)
# print(local_vars)

# Execute the definitions first
exec(definitions_code, global_vars, local_vars)

print("got here")
global_vars.update(local_vars)

exec(rest_code, global_vars, local_vars)
# print(global_vars)
print(local_vars)


custom_gvars = {}
lvars = {special_print.__name__: special_print}
out = StringIO()
with redirect_stdout(out):
    exec_after_defining(top_code_str, custom_gvars, lvars)

print(out.getvalue())