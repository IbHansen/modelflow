# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:50:07 2020

@author: bruger

Handeling of models with equiblibium condition 
for instance supply = demand 

we have to handle 
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import rc
import sys
sys.path.append('modelflow/')


import modelclass as mc
import modelsandbox as ms
import modelmanipulation as mp 

demandparam = 1,-0.5,0.0
supplyparam = 0,0.5,0.0 

fdm = '''\
             demand = demand_ofset+ demand_slope*price+ demand_second * price**2
             supply = supply_ofset+ supply_slope*price+ supply_second * price**2
<endo=price> supply = demand 
'''
fl = [f for f in fdm.split('\n')]