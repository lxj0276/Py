# -*- encoding: utf-8 -*-

# mini functions



def get_args(funcname):
    argcount = eval(funcname + '.__code__.co_argcount')
    varnames = eval(funcname + '.__code__.co_varnames')

    return varnames[:argcount]

