# -*- coding: utf-8 -*-
import modelpattern as mp 
import re
def f(a):
    return a*2
def t_to_latex(t):
    if t.var:
        var = re.sub(r'_',r'\_',t.var) 
        if t.lag:
            return f'{var}_{{t{t.lag}}}'
        else:
            return f'{var}_t'
    elif t.number:
        return t.number
    else:
        return t.op.lower()
p = mp.udtryk_parse('a+b__e+d(-1)+g(+1)+f(3) ',funks=[f])
temp = ' '.join([t_to_latex(t) for t in p])
