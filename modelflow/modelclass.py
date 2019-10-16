# -*- coding: utf-8 -*-
"""
Created on Mon Sep 02 19:41:11 2013

This module creates model class instances. 



@author: Ib
"""

from collections import defaultdict, namedtuple
from itertools import groupby,chain
import re
import pandas as pd
import sys  
import networkx as nx
import fnmatch 
import numpy as np
from itertools import chain
from collections import Counter
import time
from contextlib import contextmanager
import os
from subprocess import run 
import webbrowser as wb
import importlib
import gc
import copy 



import seaborn as sns 
from IPython.display import SVG, display, Image
try:
    from numba import jit,njit
except:
    pass
import os

import modelmanipulation as mp
import modelvis as mv
import modelpattern as pt 

# functions used in BL language 
from scipy.stats import norm 
from math import isclose,sqrt,erf 
from scipy.special import erfinv , ndtri

node = namedtuple('node','lev,parent,child')

class BaseModel():

    ''' Class which defines a model from equations
    

   The basic enduser calls are: 
   
   mi = Basemodel(equations): Will create an instance of a model
   
   And: 
       
   result = mi.xgenr(dataframe,start,end)  Will calculate a  model.
       
      In additions there are defined functions which can do useful chores. 
   
   A model instance has a number of properties among which theese can be particular useful:
   
       :allvar: Information regarding all variables 
       :basedf: A dataframe with first result created with this model instance
       :altdf: A dataframe with the last result created with this model instance 
       
   The two result dataframes are used for comparision and visualisation. The user can set both basedf and altdf.     
    
    '''

    def __init__(self, i_eq='', modelname='testmodel',silent=False,straight = False,funks=[],
                 params={},tabcomplete=True,previousbase=False,**kwargs):
        ''' initialize a model'''
        if i_eq !='':
            self.funks = funks 
            self.params = params 
            self.equations = i_eq if '$' in i_eq else mp.tofrml(i_eq,sep='\n')
            self.name = modelname
            self.straight = straight   # if True the dependency graph will not be called and calculation wil be in input sequence 
            self.save = True    # saves the dataframe in self.basedf, self.lastdf  
            self.analyzemodelnew(silent) # on board the model equations 
            self.maxstart=0
            self.genrcolumns =[]
            self.tabcomplete = tabcomplete # do we want tabcompletion (slows dovn input to large models)
            self.previousbase = previousbase # set basedf to the previous run instead of the first run 
            self.use_preorder = False    # if prolog is used in sim2d 
        return 

    def analyzemodelnew(self,silent):
        ''' Analyze a model
        
        The function creats:**Self.allvar** is a dictory with an entry for every variable in the model 
        the key is the variable name. 
        For each endogeneous variable there is a directory with thees keys:
        
        :maxlag: The max lag for this variable
        :maxlead: The max Lead for this variable
        :endo: 1 if the variable is endogeneous (ie on the left hand side of =
        :frml: String with the formular for this variable
        :frmlnumber: The number of the formular 
        :varnr: Number of this variable 
        :terms: The frml for this variable translated to terms 
        :frmlname: The frmlname for this variable 
        :startnr: Start of this variable in gauss seidel solutio vector :Advanced:
        :matrix: This lhs element is a matrix
        :dropfrml: If this frml shoud be excluded from the evaluation.
        
        
        In addition theese properties will be created: 
        
        :endogene: Set of endogeneous variable in the model 
        :exogene: Se exogeneous variable in the model 
        :maxnavlen: The longest variable name 
        :blank: An emty string which can contain the longest variable name 
        :solveorder: The order in which the model is solved - initaly the order of the equations in the model 
        
        '''
        gc.disable()
        mega = pt.model_parse(self.equations,self.funks)
        termswithvar = {t for (f,nt) in mega for t in nt if t.var}
#        varnames = list({t.var for t in termswithvar})
        termswithlag  = sorted([(t.var,'0' if t.lag == '' else t.lag) for t in termswithvar],key=lambda x : x[0])   # sorted by varname and lag  
        groupedvars   = groupby(termswithlag,key=lambda x: x[0])
        varmaxlag     = {varandlags[0] : (min([int(t[1])  for t in list(varandlags[1])]))  for varandlags in groupedvars}
        groupedvars   = groupby(termswithlag,key=lambda x: x[0])
        varmaxlead    = {varandlags[0] : (max([int(t[1])  for t in list(varandlags[1])]))  for varandlags in groupedvars}
#        self.maxlag   = min(varmaxlag[v] for v in varmaxlag.keys())
        self.maxlag   = min(v for k,v in varmaxlag.items())
        self.maxlead  = max(v for k,v in varmaxlead.items())
        self.allvar = {name: {
                            'maxlag'     : varmaxlag[name],
                            'maxlead'     : varmaxlead[name],
                            'matrix'     : 0,
#                            'startnr'    : 0,
                            'endo'       : 0} for name in {t.var for t in termswithvar} }

        #self.aequalterm = ('','','=','','')     # this is how a term with = looks like 
        self.aequalterm = ('','=','','')     # this is how a term with = looks like 
        for frmlnumber,((frml,fr,n,udtryk),nt) in enumerate(mega):
            assigpos = nt.index(self.aequalterm)                       # find the position of = 
            zendovar   = [t.var for t in nt[:assigpos] if t.var]  # variables to the left of the =
            boolmatrix = pt.kw_frml_name(n,'MATRIX')                       # do this formular define a matrix on the left of =

            for pos,endo in enumerate(zendovar):
                    if self.allvar[endo]['endo']:
                        print(' **** On the left hand side several times: ',endo )
                    self.allvar[endo]['dropfrml']   = (1 <= pos ) 
                    self.allvar[endo]['endo']       = 1
                    self.allvar[endo]['frmlnumber'] = frmlnumber                        
                    self.allvar[endo]['frml']       = frml
                    self.allvar[endo]['terms']      = nt[:]
                    self.allvar[endo]['frmlname']   = n
                    self.allvar[endo]['matrix']     = boolmatrix
                    self.allvar[endo]['assigpos']   = assigpos 
                                    
        # finished looping over all the equations     
                             
        self.endogene = {x for x in self.allvar.keys() if     self.allvar[x]['endo']}   
        self.exogene  = {x for x in self.allvar.keys() if not self.allvar[x]['endo']}

#        # the order as in the equations 
#        for iz, a in enumerate(sorted(self.allvar)):
#            self.allvar[a]['varnr'] = iz
            
        self.v_nr = sorted([(v,self.allvar[v]['frmlnumber'])  for v in self.endogene],key = lambda x:x[1]) 
        self.nrorder = [v[0] for v in self.v_nr ]
        if self.straight:                    #no sequencing 
            self.istopo = False 
            self.solveorder = self.nrorder
        else:
   
       
            try:
                self.topo = list(nx.topological_sort(self.endograph))
                self.solveorder = self.topo
                self.istopo = True
                self.solveorder = self.topo 
                # check if there is formulars with several left hand side variables 
                # this is a little tricky 
                dropvar =  [(v,self.topo.index(v),self.allvar[v]['frmlnumber']) for v in self.topo 
                             if self.allvar[v]['dropfrml']]   # all dropped vars and their index in topo and frmlnumber
                if len(dropvar):
                    multiendofrml = {frmlnr for (var,toposort,frmlnr) in dropvar} # all multi-lhs formulars 
                    dropthisvar = [v for v in self.endogene    #  theese should also be droppen, now all are dropped 
                                     if self.allvar[v]['frmlnumber'] in multiendofrml
                                               and not self.allvar[v]['dropfrml']]
                    for var in dropthisvar:
                        self.allvar[var]['dropfrml'] = True
                    
                    # now find the first lhs variable in the topo for each formulars. They have to be not dropped 
                    # this means that they should be evaluated first
                    keepthisvarnr = [min([topoindex for (var,topoindex,frmlnr) in dropvar if frmlnr == thisfrml])
                                     for thisfrml in multiendofrml]
                    keepthisvar = [self.topo[nr] for nr in keepthisvarnr]
                    for var in keepthisvar:
                        self.allvar[var]['dropfrml'] = False
         
            except:
                print('This model has simultaneous elements or cyclical elements. The formulars will be evaluated in input sequence')
                self.istopo = False 
                self.solveorder = self.nrorder

        # for pretty printing of variables
        self.maxnavlen = max([len(a) for a in self.allvar.keys()])
        self.blank = ' ' * (self.maxnavlen + 9)  # a blank of right lenth
        gc.enable()
#        gc.collect()
        return
    
    def smpl(self,start='',slut='',df=None):
        ''' Defines the model.current_per which is used for calculation period/index
        when no parameters are issues the current current period is returned \n
        Either none or all parameters have to be provided '''
        df_ = self.basedf if df is None else df
        if start =='' and slut == '':
            if not hasattr(self,'current_per'):  # if first invocation just use the max slize 
                istart,islut= df_.index.slice_locs(df_.index[0-self.maxlag],df_.index[-1-self.maxlead],kind='loc')
                self.current_per=df_.index[istart:islut]
        elif start ==0 and slut == 0:
                istart,islut= df_.index.slice_locs(df_.index[0-self.maxlag],df_.index[-1],kind='loc')
                self.current_per=df_.index[istart:islut]
        else:
            istart,islut= df_.index.slice_locs(start,slut,kind='loc')
            per=df_.index[istart:islut]
            self.current_per =  per 
        self.old_current_per = copy.deepcopy(self.current_per)
        return self.current_per

    @property
    def reset_smpl():
        '''Reset the smpl to previous value''' 
        self.current_per = self.old_current_per

    @property
    def push_smpl():
        '''pusch the Reset the smpl to previous value''' 
        self.pushed_current_per = copy.deepcopy(self.current_per)
        
    @property
    def pull_smpl():
        '''pull the pushed smpl''' 
        self.current_per = copy.deepcopy(self.pushed_current_per)

    @property
    def endograph(self) :
        ''' Dependencygraph for currrent periode endogeneous variable, used for reorder the equations'''
        if not hasattr(self,'_endograph'):
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() if  inf['endo'])
            
            rhss    = ((var,term[self.allvar[var]['assigpos']:]) for  var,term in terms )
            rhsvar = ((var,{v.var for v in rhs if v.var and v.var in self.endogene and v.var != var and not v.lag}) for var,rhs in rhss)
            
            edges = ((v,e) for e,rhs in rhsvar for v in rhs)
#            print(edges)
            self._endograph=nx.DiGraph(edges)
            self._endograph.add_nodes_from(self.endogene)
        return self._endograph  
  

    @property 
    def calculate_freq(self):
        ''' The number of operators in the model '''
        if not hasattr(self,'_calculate_freq'):
            operators = ( t.op  for v in self.endogene for t in self.allvar[v]['terms'] if (not self.allvar[v]['dropfrml']) and t.op and  t.op not in '$,()=[]'  )
            res = Counter(operators).most_common()
            all = sum((n[1] for n in res))
            self._calculate_freq = res+[('Total',all)]
        return self._calculate_freq 
        

    def get_columnsnr(self,df):
        ''' returns a dict a databanks variables as keys and column number as item
        used for fast getting and setting of variable values in the dataframe'''
        return {v: i for i,v in enumerate(df.columns) }
   
        
    def outeval(self,databank):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '

        def totext(t):
            ''' This function returns a python representation of a term'''
            if t.op:
                return  '\n' if ( t.op == '$' ) else t.op.lower()
            elif t.number:
                return  t.number
            elif t.var:
                return 'values[row'+t.lag+','+str(columnsnr[t.var])+']' 
                
        columnsnr=self.get_columnsnr(databank)
        fib1 =     ['def make_los(funks=[]):\n']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(values,row,solveorder, allvar):\n')
        fib1.append(long+'try :\n')
        startline = len(fib1)+1
        content = (longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                   else ''.join( (totext(t) for t in self.allvar[v]['terms'])) )   
                       for v in self.solveorder )
           
        fib2 =    [long+ 'except :\n']
        fib2.append(longer + 'print("Error in",allvar[solveorder[sys.exc_info()[2].tb_lineno-'+str(startline)+']]["frml"])\n')
        fib2.append(longer + 'raise\n')
        fib2.append(long + 'return \n')
        fib2.append(short + 'return los\n')
        return ''.join(chain(fib1,content,fib2))   
        
    def errfunk(self,linenr,startlines=4):
        ''' developement function
        
        to handle run time errors in model calculations'''
        
        winsound.Beep(500,1000)
        print('>> Error in     :',self.name)
        print('>> At           :',self._per)
        self.errdump = pd.DataFrame(self.values,columns=self.currentdf.columns, index= self.currentdf.index)
        outeq = self.currentmodel[linenr-startlines] 
        varout = sorted(list({(var,lag) for (var,lag) in re.findall(self.ypat,outeq) if var not in self.funk}))
        print('>> Equation     :',outeq)
        maxlen = str(3+max([len(var) for (var,lag) in varout])) 
        fmt = '{:>'+maxlen+'} {:>3} {:>20} '
        print('>>',fmt.format('Var','Lag','Value'))
        for var,lag in varout: 
            lagout = 0 if lag =='' else int(lag)
            print('>>',('{:>'+maxlen+'} {:>3} {:>20} ').format(var,lagout,self.errdump.loc[self._per+lagout,var]))
        print('A snapshot of the data at the error point is at .errdump ')

    def eqcolumns(self,a,b):
        ''' compares two lists '''
        if len(a)!=len(b):
            return False
        else:
            return all(a == b)
             
    def xgenr(self, databank, start='', slut='', silent=0,samedata=1,**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
 
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
#        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
        if (not samedata) or (not hasattr(self,'solve_dag')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            self.genrcolumns = databank.columns.copy()  
            make_los_text =  self.outeval(databank)
            self.make_los_text = make_los_text
            exec(make_los_text,globals())  # creates the los function
            self.solve_dag  = make_los(self.funks)
        values = databank.values.copy()  # 
        for periode in sol_periode:
            row=databank.index.get_loc(periode)
            self.solve_dag(values, row , self.solveorder , self.allvar)
            if not silent : print (periode, ' solved')
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)    
        if not silent : print (self.name + ' calculated ')
        return outdf 
  
    def findpos(self):
        ''' find a startposition in the calculation array for a model 
        places startposition for each variable in model.allvar[variable]['startpos']
        places the max startposition in model.maxstart ''' 
        
        if self.maxstart == 0      :   
            variabler=(x for x in sorted(self.allvar.keys()))
            start=0
            for v,m in ((v,self.allvar[v]['maxlag']) for v in variabler):
                self.allvar[v]['startnr']=start
                start=start+(-int(m))+1
            self.maxstart=start   # print(v.ljust(self.maxnavlen),str(m).rjust(6),str(self.allvar[v]['start']).rju

    def make_gaussline(self,vx,nodamp=False):
        ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
        simultanius models
        the variables are mapped to position in a vector which has all relevant varaibles lagged 
        this is in order to provide opertunity to optimise data and solving 
        
        New version to take hand of several lhs variables. Dampning is not allowed for
        this. But can easely be implemented by makeing a function to multiply tupels
        '''
        termer=self.allvar[vx]['terms']
        assigpos =  self.allvar[vx]['assigpos'] 
        if nodamp:
            ldamp=False
        else:     
            if 'Z' in self.allvar[vx]['frmlname']: # convention for damping equations 
                assert assigpos == 1 , 'You can not dampen equations with several left hand sides:'+vx
                endovar=[t.op if t.op else ('a['+str(self.allvar[t.var]['startnr'])+']') for j,t in enumerate(termer) if j <= assigpos-1 ]
                damp='(1-alfa)*('+''.join(endovar)+')+alfa*('      # to implemet dampning of solution
                ldamp = True
            else:
                ldamp = False
        out=[]
        
        for i,t in enumerate(termer[:-1]): # drop the trailing $
            if t.op:
                out.append(t.op.lower())
                if i == assigpos and ldamp:
                    out.append(damp)
            if t.number:
                out.append(t.number)
            elif t.var:
                lag=int(t.lag) if t.lag else 0
                out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')              
        if ldamp: out.append(')') # the last ) in the dampening 
        res = ''.join(out)
        return res

    def make_resline(self,vx):
        ''' takes a list of terms and translates to a line calculating linne
        '''
        termer=self.allvar[vx]['terms']
        assigpos =  self.allvar[vx]['assigpos'] 
        out=[]
        
        for i,t in enumerate(termer[:-1]): # drop the trailing $
            if t.op:
                out.append(t.op.lower())
            if t.number:
                out.append(t.number)
            elif t.var:
                lag=int(t.lag) if t.lag else 0
                if i < assigpos:
                    out.append('b['+str(self.allvar[t.var]['startnr']-lag)+']')              
                else:
                    out.append('a['+str(self.allvar[t.var]['startnr']-lag)+']')              
        res = ''.join(out)
        return res

    def createstuff3(self,dfxx):
        ''' Connect a dataframe with the solution vector used by the iterative sim2 solver) 
        return a function to place data in solution vector and to retrieve it again. ''' 
    
        columsnr =  {v: i for i,v in enumerate(dfxx.columns) }
        pos0 = sorted([(self.allvar[var]['startnr']-lag,(var,lag,columsnr[var]) ) 
            for var in self.allvar for lag  in range(0,-1+int(self.allvar[var]['maxlag']),-1)])
#      if problems check if find_pos has been calculated 
        posrow = np.array([lag for (startpos,(var,lag,colpos)) in pos0 ])
        poscol = np.array([colpos for (startpos,(var,lag,colpos)) in pos0 ])
        
        poscolendo   = [columsnr[var]  for var in self.endogene ]
        posstartendo = [self.allvar[var]['startnr'] for var in self.endogene ]
        
        def stuff3(values,row,ljit=False):
            '''Fills  a calculating vector with data, 
            speeded up by using dataframe.values '''
            
            if ljit:
#                a = np.array(values[posrow+row,poscol],dtype=np.dtype('f8'))
#                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))
                a = np.ascontiguousarray(values[posrow+row,poscol],dtype=np.dtype('f8'))
            else:
                a = values[posrow+row,poscol]
            return a            
        
        def saveeval3(values,row,vector):
            values[row,poscolendo] = vector[posstartendo]

        return stuff3,saveeval3 
    
    
   
    def outsolve(self,order='',exclude=[]):
        ''' returns a string with a function which calculates a 
        Gauss-Seidle iteration of a model
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        model.allvar[v]["gauss"] the ccalculation 
        '''
        short,long,longer = 4*' ',8*' ',12 *' '
        solveorder=order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower())
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(a,alfa):')
        f2=(long + self.make_gaussline(v) for v in solveorder 
              if (v not in exclude) and (not self.allvar[v]['dropfrml']))
        fib2 = [long + 'return a ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1,f2,fib2))
        return out
    def make_solver(self,ljit=False,order='',exclude=[],cache=False):
        ''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
        model.solve 
        model.solve_jit '''
        
        a=self.outsolve(order,exclude) # find the text of the solve
#        if cache:
#            with open('xxsolver.py','wt') as out:
#                out.write(a)
#            import xxsolver
#            make= xxsolver.make
#            #xx = importlib.import_module('\\python poc - subclass\\xxsolver')  #
#        else:
        exec(a,globals()) # make the factory defines
        self.solve=make(funks=self.funks) # using the factory create the function 
        if ljit:
            print('Time for a cup of coffee')
            self.solve_jit=jit("f8[:](f8[:],f8)",cache=cache,fastmath=True)(self.solve)
        return 


    def sim(self,databank,start='',slut='', antal=1,first_test=1,ljit=False,exclude=[],silent=False,new=False,
             conv=[],samedata=True,dumpvar=[],ldumpvar=False,
             dumpwith=15,dumpdecimal=5,lcython=False,setbase=False,
             setlast=True,alfa=0.2,sim=True,absconv=0.01,relconv=0.00001,
             debug=False,stats=False,**kwargs):
        ''' solves a model with data from a databank if the model has a solve function else it will be created.
        
        The default options are resonable for most use:
        
        :start,slut: Start and end of simulation, default as much as possible taking max lag into acount  
        :antal: Max interations
        :first_test: First iteration where convergence is tested
        :ljit: If True Numba is used to compile just in time - takes time but speeds solving up 
        :new: Force creation a new version of the solver (for testing)
        :exclude: Don't use use theese foormulas
        :silent: Suppres solving informations 
        :conv: Variables on which to measure if convergence has been achived 
        :samedata: If False force a remap of datatrframe to solving vector (for testing) 
        :dumpvar: Variables to dump 
        :ldumpvar: toggels dumping of dumpvar 
        :dumpwith: with of dumps
        :dumpdecimal: decimals in dumps 
        :lcython: Use Cython to compile the model (experimental )
        :alfa: Dampning of formulas marked for dampning (<Z> in frml name)
        :sim: For later use
        :absconv: Treshold for applying relconv to test convergence 
        :relconv: Test for convergence 
        :debug:  Output debug information 
        :stats:  Output solving statistics
        
        
       '''
        
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** ERROR: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        self.findpos()
        databank = insertModelVar(databank,self)   # fill all Missing value with 0.0 
   
        with ttimer('create stuffer and gauss lines ',debug) as t:        
            if (not hasattr(self,'stuff3')) or  (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3,self.saveeval3  = self.createstuff3(databank)
                self.simcolumns=databank.columns.copy()
                
        with ttimer('Create solver function',debug) as t: 
            if ljit:
                   if not hasattr(self,'solve_jit'): self.make_solver(ljit=True,exclude=exclude)
                   this_solve = self.solve_jit
            elif lcython:
                    if not hasattr(self,'solve_cyt'):
                        from  slos import los 
                        self.solve_cyt = los
                    this_solve = self.solve_cyt 
            else:  
                    if not hasattr(self,'solve'): self.make_solver(exclude=exclude)
                    this_solve = self.solve                

        values=databank.values.copy()
#        columsnr=self.get_columnsnr(databank)
        ittotal=0 # total iteration counter 
        convvar = [conv] if isinstance(conv,str) else conv if conv != [] else list(self.endogene)  
        convplace=[self.allvar[c]['startnr'] for c in convvar] # this is how convergence is measured  
        if ldumpvar:
            dump = convvar if dumpvar == [] else self.vlist(dumpvar)
            dumpplac = [self.allvar[v]['startnr'] for v in dump]
            dumphead = ' '.join([('{:>'+str(dumpwith)+'}').format(d) for d in dump])
        starttime=time.time()
        for periode in sol_periode:
            row=databank.index.get_loc(periode)
            with ttimer('stuffing',debug) as tt:
                a=self.stuff3(values,row,ljit)
#                b=self.stuff2(values,row,columsnr)
#                assert all(a == b)

            if ldumpvar:
                print('\nStart solving',periode)
                print('             '+dumphead)
                print('Start        '+' '.join([('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))
            jjj=0

            for j in range(antal):
                jjj=j+1
                if debug :print('iteration :',j)
                with ttimer('iteration '+str(jjj),debug) as tttt:
                    itbefore=a[convplace].copy()
                    a=this_solve(a,alfa) 
                if ldumpvar: print('Iteration {:>3}'.format(j)+' '.join([('{:>'+str(dumpwith)+',.'+str(dumpdecimal)+'f}').format(a[p]) for p in dumpplac]))

                if j > first_test: 
                    itafter=a[convplace].copy()
                    convergence = True
                    for after,before in zip(itafter,itbefore):
#                        print(before,after)
                        if before > absconv and abs(after-before)/abs(before)  > relconv:
                            convergence = False
                            break 
                    if convergence:
                        if not silent: print(periode,'Solved in ',j,'iterations')
                        break
                    else:
                        itbefore=itafter.copy()
            else:
                print('No convergence ',periode,' after',jjj,' iterations')
            with ttimer('saving',debug) as t:            
#                self.saveeval2(values,row,columsnr,a) # save the result 
                self.saveeval3(values,row,a) # save the result 
            ittotal =ittotal+jjj
        if not silent : print(self.name,': Solving finish from ',sol_periode[0],'to',sol_periode[-1])
        outdf  =  pd.DataFrame(values,index=databank.index,columns=databank.columns)
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            simtime = endtime-starttime
            print('{:<40}:  {:>15,}'.format('Floating point operations :',self.calculate_freq[-1][1]))
            print('{:<40}:  {:>15,}'.format('Total iterations :',ittotal))
            print('{:<40}:  {:>15,}'.format('Total floating point operations',numberfloats))
            print('{:<40}:  {:>15,.2f}'.format('Simulation time (seconds) ',simtime))
            if simtime > 0.0:
                print('{:<40}:  {:>15,.0f}'.format('Floating point operations per second',numberfloats/simtime))
        return outdf 
    
  
    def __len__(self):
        return len(self.endogene)
    
    def __repr__(self):
        fmt = '{:40}: {:>20} \n'
        out = fmt.format('Model name',self.name)
        out += fmt.format('Model structure ', 'Recursive' if  self.istopo else 'Simultaneous') 
        out += fmt.format('Number of variables ',len(self.allvar))
        out += fmt.format('Number of exogeneous  variables ',len(self.exogene))
        out += fmt.format('Number of endogeneous variables ',len(self.endogene))
        return '<\n'+out+'>'
        
def calc(df,expressions,start=False,end=False,silent=True):
    ''' A function, wich calculates expressions seperated by ';' or linebreaks 
    if no start is specified the max lag will be used ''' 
    eq = '\n'.join(['FRML <> ' + e +' $' for e in expressions.replace(';','\n').strip().split('\n')])
    mmodel = BaseModel(eq)
    tstart = df.index.get_loc(start) if start else -mmodel.maxlag
    tend   = df.index.get_loc(end) if end else -1
    out = mmodel.xgenr(df,start=df.index[tstart],slut=df.index[tend],silent=silent)  
    return out
        
# 

class model(BaseModel):
    ''' The model class, used for calculating models
    
    Compared to BaseModel it allows for simultaneous model and contains a number of properties 
    and functions to analyze and manipulate models and visualize results.  
    
    '''
    def __call__(self, *args, **kwargs ):
        ''' Runs a model. 
        
        Default a straight model is calculated by *xgenr* a simultaneous model is solved by *sim* 
        
        :sim: If False forces a  model to be calculated (not solved) if True force simulation 
        :setbase: If True, place the result in model.basedf 
        :setlast: if False don't place the results in model.lastdf
        
        if the modelproperty previousbase is true, the previous run is used as basedf. 

        
        '''
        if self.save:
            if self.previousbase and hasattr(self,'lastdf'):
                self.basedf = self.lastdf.copy(deep=True)
            
        if 'sim' in kwargs:
            outdf = self.sim(*args, **kwargs ) if kwargs['sim'] else self.xgenr( *args, **kwargs)   
        else: 
            outdf = self.xgenr( *args, **kwargs) if (self.straight or self.istopo) else self.sim(*args, **kwargs ) # sim2 makes its own saving

        if self.save:
            if (not hasattr(self,'basedf')) or kwargs.get('setbase',False) : self.basedf = outdf.copy(deep=True) 
            if kwargs.get('setlast',True)                                  : self.lastdf = outdf.copy(deep=True)
    
        return outdf

    @property
    def lister(self):
        return pt.list_extract(self.equations)   # lists used in the equations 


    @property
    def listud(self):
        '''returns a string of the models listdefinitions \n
        used when ceating (small) models based on this model '''
        udlist=[]
        for l in self.lister: 
            udlist.append('list '+l+' =  ')
            for j,sl in enumerate(self.lister[l]):
                lensl=str(max(30,len(sl)))
                if j >= 1 : udlist.append(' / ')
                udlist.append(('{:<'+lensl+'}').format('\n    '+sl+' '))
                udlist.append(':')
                for i in self.lister[l][sl]:
                    udlist.append(' '+i)

            udlist.append(' $ \n')
        return ''.join(udlist)
    
    def vlist(self,pat):
        '''returns a list of variable in the model matching the pattern, the pattern can be a list of patterns'''
        if isinstance(pat,list):
               upat=pat
        else:
               upat = [pat]
               
        ipat = upat
            
        try:       
            out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(self.allvar.keys(),up.upper()))]  
        except:
            ''' in case the model instance is an empty instance around datatframes, typical for visualization'''
            out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(self.lastdf.columns,up.upper()))]  
        return out
    
    
    def write_eq(self,name='My_model.fru',lf=False):
        ''' writes the formulas to file, can be input into model 

        lf=True -> new lines are put after each frml ''' 
        with open(name,'w') as out:
            outfrml = self.equations.replace('$','$\n') if lf else self.equations 
                
            out.write(outfrml)

        
    def print_eq(self, varnavn, data='', start='', slut=''):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period'''
        print_per=self.smpl(start, slut,data)
        minliste = [(term.var, term.lag if term.lag else '0')
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print (self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        print(''.join(['{:>20}'.format(str(per)) for per in print_per]))

        for var,lag in sorted(set(minliste), key=lambda x: minliste.index(x)):
            endoexo='E' if self.allvar[var]['endo'] else 'X'
            print(endoexo+': {0:50}{1:>5}'.format(var, lag), end='')
            print(''.join(['{:>20.4f}'.format(data.loc[per+int(lag),var])  for per in print_per]))
        print('\n')
        return 
        
        
    def print_eq_mul(self, varnavn, grund='',mul='', start='', slut='',impact=False):
        #from pandabank import get_var
        '''Print all variables that determines input variable (varnavn)
          optional -- enter period and databank to get var values for chosen period'''
        grund.smpl(start, slut)
        minliste = [[term.var, term.lag if term.lag else '0']
                    for term in self.allvar[varnavn]['terms'] if term.var]
        print (self.allvar[varnavn]['frml'])
        print('{0:50}{1:>5}'.format('Variabel', 'Lag'), end='')
        for per in grund.current_per:
            per = str(per)
            print('{:>20}'.format(per), end='')
        print('')
        diff=mul.data-grund.data
        endo=minliste[0]
        for item in minliste:
            target_column = diff.columns.get_loc(item[0])
            print('{0:50}{1:>5}'.format(item[0], item[1]), end='')
            for per in grund.current_per:
                target_index = diff.index.get_loc(per) + int(item[1])
                tal=diff.iloc[target_index, target_column]
                tal2=tal*self.diffvalue_d3d if impact else 1
                print('{:>20.4f}'.format(tal2 ), end='')
            print(' ')

    def print_all_equations(self, inputdata, start, slut):
        '''Print values and formulas for alle equations in the model, based input database and period \n
        Example: stress.print_all_equations(bankdata,'2013Q3')'''
        for var in self.solveorder:
            # if var.find('DANSKE')<>-2:
            self.print_eq(var, inputdata, start, slut)
            print ('\n' * 3)

    def print_lister(self):
        ''' prints the lists used in defining the model ''' 
        for i in self.lister:
            print(i)
            for j in self.lister[i]:
                print(' ' * 5 , j , '\n' , ' ' * 10, [xx for xx in self.lister[i][j]])
                
    @property 
    def strongorder(self):
        if not hasattr(self,'_strongorder'):
               self._strongorder = create_strong_network(self.endograph)
        return self._strongorder
    
    @property 
    def strongblock(self):
        if not hasattr(self,'_strongblock'):
                xx,self._strongblock,self._strongtype = create_strong_network(self.endograph,typeout=True)
        return self._strongblock
    
    @property 
    def strongtype(self):
        if not hasattr(self,'_strongtype'):
                xx,self._strongblock,self._strongtype = create_strong_network(self.endograph,typeout=True)
        return self._strongtype

    @property 
    def strongfrml(self):
        ''' To search simultaneity (circularity) in a model 
        this function returns the equations in each strong block
        
        '''
        simul = [block for block,type in zip(self.strongblock,self.strongtype) if type.startswith('Simultaneous') ]
        out = '\n\n'.join(['\n'.join([self.allvar[v]['frml'] for v in block]) for block in simul])
        return 'Equations with feedback in this model:\n'+out
    
    def superblock(self):
        """ finds prolog, core and epilog variables """

        if not hasattr(self,'_prevar'):
            self._prevar = []
            self._epivar = []
            this = self.endograph.copy()
            
            while True:
                new = [v for v,indegree in this.in_degree() if indegree==0]
                if len(new) == 0:
                    break
                self._prevar = self._prevar+new
                this.remove_nodes_from(new)
                
            while True:
                new = [v for v,outdegree in this.out_degree() if outdegree==0]
                if len(new) == 0:
                    break
                self._epivar = new + self._epivar
                this.remove_nodes_from(new)
                
            episet = set(self._epivar)
            preset = set(self._prevar)
            self.common_pre_epi_set = episet.intersection(preset)
            noncore =  episet  | preset 
            self._coreorder = [v for v in self.nrorder if not v in noncore]

            xx,self._corestrongblock,self._corestrongtype = create_strong_network(this,typeout=True)
            self._superstrongblock        = ([self._prevar] + 
                                            (self._corestrongblock if  len(self._corestrongblock) else [[]]) 
                                            + [self._epivar])
            self._superstrongtype         = ( ['Recursiv'] +   
                                            (self._corestrongtype  if  len(self._corestrongtype)  else [[]])
                                            + ['Recursiv'] )
            self._corevar = list(chain.from_iterable((v for v in self.nrorder if v in block) for block in self._corestrongblock)
                                )
            

    
    @property 
    def prevar(self):
        """ returns a set with names of endogenopus variables which do not depend 
        on current endogenous variables """

        if not hasattr(self,'_prevar'):
            self.superblock()
        return self._prevar
    
    @property 
    def epivar(self):
        """ returns a set with names of endogenopus variables which do not influence 
        current endogenous variables """

        if not hasattr(self,'_epivar'):
            self.superblock()
        return self._epivar
    
    @property
    def preorder(self):
        ''' the endogenous variables which can be calculated in advance '''
#        return [v for v in self.nrorder if v in self.prevar]
        return self.prevar
    
    @property
    def epiorder(self):
        ''' the endogenous variables which can be calculated in advance '''
        return self.epivar
    
    @property
    def coreorder(self):
        ''' the endogenous variables which can be calculated in advance '''
        if not hasattr(self,'_coreorder'):    
            self.superblock()
        return self._corevar 

     
    @property
    def precoreepiorder(self):
        return self.preorder+self.coreorder+self.epiorder 

    @property 
    def prune_endograph(self):
        if not hasattr(self,'_endograph'):
            _ = self.endograph
        self._endograph.remove_nodes_from(self.prevar)
        self._endograph.remove_nodes_from(self.epivar)
        return self._endograph 

    @property
    def use_preorder(self):
        return self._use_preorder
    
    @use_preorder.setter
    def use_preorder(self,use_preorder):
        if use_preorder:
            if self.istopo or self.straight:
                print(f"You can't use preorder in this model, it is topological or straight")
                print(f"We pretend you did not try to set the option")
            else:
                self._use_preorder = True
                self._oldsolveorder = self.solveorder[:]
                self.solveorder = self.precoreepiorder
        else:
            self._use_preorder = False
            if hasattr(self,'_oldsolveorder'):
                self.solveorder = self._oldsolveorder
        
        
     
    @property 
    def totgraph_nolag(self):
        
        ''' The graph of all variables, lagged variables condensed'''
        if not hasattr(self,'_totgraph_nolag'):
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() 
                           if  inf['endo'])
            
            rhss    = ((var,term[term.index(self.aequalterm):]) for  var,term in terms )
            rhsvar = ((var,{v.var for v in rhs if v.var and v.var != var }) for var,rhs in rhss)
            
    #            print(list(rhsvar))
            edges = (((v,e) for e,rhs in rhsvar for v in rhs))
            self._totgraph_nolag = nx.DiGraph(edges)
        return self._totgraph_nolag  
    
    @property 
    def totgraph(self):
        ''' The graph of all variables including and seperate lagged variable '''
        if not hasattr(self,'_totgraph'):

            def lagvar(xlag):
                ''' makes a string with lag ''' 
                return   '('+str(xlag)+')' if int(xlag) < 0 else '' 
            
            terms = ((var,inf['terms']) for  var,inf in self.allvar.items() 
                           if  inf['endo'])
            
            rhss    = ((var,term[term.index(self.aequalterm):]) for  var,term in terms )
            rhsvar = ((var,{(v.var+'('+v.lag+')' if v.lag else v.var) for v in rhs if v.var}) for var,rhs in rhss)
            
    #            print(list(rhsvar))
            edges = (((v,e) for e,rhs in rhsvar for v in rhs))
            edgeslag = [(v+lagvar(lag+1),v+lagvar(lag)) for v,inf in self.allvar.items() for lag in range(inf['maxlag'],0)]
    #        edgeslag = [(v,v+lagvar(lag)) for v,inf in m2test.allvar.items() for lag in range(inf['maxlag'],0)]
            self._totgraph = nx.DiGraph(chain(edges,edgeslag))

        return self._totgraph             


    def graph_remove(self,paralist):
        ''' Removes a list of variables from the totgraph and totgraph_nolag 
        mostly used to remove parmeters from the graph, makes it less crowded'''
        
        if not hasattr(self,'_totgraph') or not hasattr(self,'_totgraph_nolag'):
            _ = self.totgraph
            _ = self.totgraph_nolag
        
        self._totgraph.remove_nodes_from(paralist)
        self._totgraph_nolag.remove_edges_from(paralist)        
        return 

    def graph_restore(self):
        ''' If nodes has been removed by the graph_remove, calling this function will restore them '''
        if hasattr(self,'_totgraph') or hasattr(self,'_totgraph_nolag'):
            delattr(self,'_totgraph')
            delattr(self,'_totgrapH_nolag')        
        return 

    
    def exodif(self,a=None,b=None):
        ''' Finds the differences between two dataframes in exogeneous variables for the model
        Defaults to getting the two dataframes (basedf and lastdf) internal to the model instance ''' 
        aexo=a.loc[:,self.exogene] if isinstance(a,pd.DataFrame) else self.basedf.loc[:,self.exogene]
        bexo=b.loc[:,self.exogene] if isinstance(b,pd.DataFrame) else self.lastdf.loc[:,self.exogene] 
        diff = pd.eval('bexo-aexo')
        out2=diff.loc[(diff != 0.0).any(axis=1),(diff != 0.0).any(axis=0)]
                     
        return out2.T.sort_index(axis=0).T    
    
    
    def outres(self,order='',exclude=[]):
        ''' returns a string with a function which calculates a 
        calculation for residual check 
        exclude is list of endogeneous variables not to be solved 
        uses: 
        model.solveorder the order in which the variables is calculated
        '''
        short,long,longer = 4*' ',8*' ',12 *' '
        solveorder=order if order else self.solveorder
        fib1 = ['def make(funks=[]):']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower())
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower())
        fib1.append(short + 'from numpy import zeros,float64')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        fib1.append(short + 'def los(a):')
        fib1.append(long+'b=zeros(len(a),dtype=float64)\n')
        f2=[long + self.make_resline(v) for v in solveorder 
              if (v not in exclude) and (not self.allvar[v]['dropfrml'])]
        fib2 = [long + 'return b ']
        fib2.append(short+'return los')
        out = '\n'.join(chain(fib1,f2,fib2))
        return out
    
       
    def make_res(self,order='',exclude=[]):
        ''' makes a function which performs a Gaus-Seidle iteration
        if ljit=True a Jittet function will also be created.
        The functions will be placed in: 
        model.solve 
        model.solve_jit '''
        
        xxx=self.outres(order,exclude) # find the text of the solve
        exec(xxx,globals()) # make the factory defines
        res_calc = make(funks=self.funks) # using the factory create the function 
        return res_calc
    
    def res(self,databank,start='',slut='',silent=1):
        ''' calculates a model with data from a databank
        Used for check wether each equation gives the same result as in the original databank'
        '''
        if not hasattr(self,'res_calc'):
            self.findpos()
            self.res_calc = self.make_res() 
        databank=insertModelVar(databank,self)   # kan man det her? I b 
        values=databank.values
        bvalues=values.copy()
        sol_periode = self.smpl(start,slut,databank)
        stuff3,saveeval3  = self.createstuff3(databank)
        for per in sol_periode:
            row=databank.index.get_loc(per)
            aaaa=stuff3(values,row)
            b=self.res_calc(aaaa) 
            if not silent: print(per,'Calculated')
            saveeval3(bvalues,row,b)
        xxxx =  pd.DataFrame(bvalues,index=databank.index,columns=databank.columns)    
        if not silent: print(self.name,': Res calculation finish from ',sol_periode[0],'to',sol_periode[-1])
        return xxxx 

    
    def get_att_pct(self,n,filter = True,lag=True,start='',end=''):
        ''' det attribution pct for a variable.
         I little effort to change from multiindex to single node name''' 
        res = self.dekomp(n,lprint=0,start=start,end=end)
        res_pct = res[2].iloc[:-2,:]
        if lag:
            out_pct = pd.DataFrame(res_pct.values,columns=res_pct.columns,
                 index=[r[0]+(f'({str(r[1])})' if  r[1] else '') for r in res_pct.index])
        else:
            out_pct = res_pct.groupby(level=[0]).sum()
        out = out_pct.loc[(out_pct != 0.0).any(axis=1),:] if filter else out_pct
        return out
    
    
    def get_eq_values(self,varnavn,last=True,databank=None,nolag=False,per=None,showvar=False,alsoendo=False):
        ''' Returns a dataframe with values from a frml determining a variable 
        
        
         options: 
             :last:  the lastdf is used else baseline dataframe
             :nolag:  only line for each variable ''' 
        
        if varnavn in self.endogene: 
            if type(databank)==type(None):
                df=self.lastdf if last else self.basedf 
            else:
                df=databank
                
            if per == None :
                current_per = self.current_per
            else:
                current_per = per 
                
            varterms     = [(term.var, int(term.lag) if term.lag else 0)
#                            for term in self.allvar[varnavn.upper()]['terms'] if term.var]
                            for term in self.allvar[varnavn.upper()]['terms'] if term.var and not (term.var ==varnavn.upper() and term.lag == '')]
            sterms = sorted(set(varterms),key= lambda x: (x[0],-x[1])) # now we have droped dublicate terms and sorted 
            if nolag: 
                sterms = sorted({(v,0) for v,l in sterms})
            if showvar: sterms = [(varnavn,0)]+sterms    
            lines = [[get_a_value(df,p,v,lag) for p in current_per] for v,lag in sterms]
            out = pd.DataFrame(lines,columns=current_per,
                   index=[r[0]+(f'({str(r[1])})' if  r[1] else '') for r in sterms])
            return out
        else: 
            return None 

    def print_eq_values(self,varname,databank=None,all=False,dec=1,lprint=1,per=None):
        ''' for an endogeneous variable, this function prints out the frml and input variale
        for each periode in the current_per. 
        The function takes special acount of dataframes and series '''
        res = self.get_eq_values(varname,showvar=True,databank=databank,per=per)
        out = ''
        if type(res) != type(None):
            varlist = res.index.tolist()
            maxlen = max(len(v) for v in varlist)
            out +=f'\nCalculations of {varname} \n{self.allvar[varname]["frml"]}'
            for per in res.columns:
                out+=f'\n\nLooking at period:{per}'
                for v in varlist:
                    this = res.loc[v,per]
                    if type(this) == pd.DataFrame:
                        vv = this if all else this.loc[(this != 0.0).any(axis=1),(this != 0.0).any(axis=0)]
                        out+=f'\n: {v:{maxlen}} = \n{vv.to_string()}\n'
                    elif type(this) == pd.Series:
                        ff = this.astype('float') 
                        vv = ff if all else ff.iloc[ff.nonzero()[0]]
                        out+=f'\n{v:{maxlen}} = \n{ff.to_string()}\n'
                    else:
                        out+=f'\n{v:{maxlen}} = {this:>20}'
                        
            if lprint:
                print(out)
            else: 
                return out 
            
    def print_all_eq_values(self,databank=None,dec=1):
        for v in self.solveorder:
            self.print_eq_values(v,databank,dec=dec)
            
    def get_eq_dif(self,varnavn,filter=False,nolag=False,showvar=False) :
        ''' returns a dataframe with difference of values from formula'''
        out0 =  (self.get_eq_values(varnavn,last=True,nolag=nolag,showvar=showvar)-
                 self.get_eq_values(varnavn,last=False,nolag=nolag,showvar=showvar))
        if filter:
            mask = out0.abs()>=0.00000001
            out = out0.loc[mask] 
        else:
            out=out0
        return out 


    def get_values(self,v): 
        ''' returns a dataframe with the data points for a node,  including lags ''' 
        t = pt.udtryk_parse(v,funks=[])
        var=t[0].var
        lag=int(t[0].lag) if t[0].lag else 0
        bvalues = [float(get_a_value(self.basedf,per,var,lag)) for per in self.current_per] 
        lvalues = [float(get_a_value(self.lastdf,per,var,lag)) for per in self.current_per] 
        dvalues = [float(get_a_value(self.lastdf,per,var,lag)-get_a_value(self.basedf,per,var,lag)) for per in self.current_per] 
        df = pd.DataFrame([bvalues,lvalues,dvalues],index=['Base','Last','Diff'],columns=self.current_per)
        return df 
    
    def __getitem__(self, name):
        
        a=self.vis(name)
        return a
    
    def __getattr__(self, name):
        try:
            return mv.varvis(model=self,var=name.upper())
        except:
#            print(name)
            raise AttributeError 
            pass                
    
       
    def __dir__(self):
        if self.tabcomplete:
            if not hasattr(self,'_res'):
                self._res = sorted(list(self.allvar.keys()) + list(self.__dict__.keys()) + list(type(self).__dict__.keys()))
            return self. _res  

        else:       
            res = list(self.__dict__.keys())
        return res


    def dekomp(self, varnavn, start='', end='',basedf=None,altdf=None,lprint=True):
        '''Print all variables that determines input variable (varnavn)
        optional -- enter period and databank to get var values for chosen period'''
       
    #    model=mtotal
    #    altdf=adverse
    #    basedf=base
    #    start='2016q1'
    #    end = '2016q4'
    #    varnavn = 'LOGITPD__FF_NFC_NONRE__DE'
        basedf_ = basedf if isinstance( basedf,pd.DataFrame) else self.basedf
        altdf_  = altdf  if isinstance( altdf,pd.DataFrame) else self.lastdf 
        start_  = start  if start != '' else self.current_per[0]
        end_    = end    if end   != '' else self.current_per[-1]
        
    
        mfrml        = model(self.allvar[varnavn]['frml'],funks=self.funks)   # calculate the formular 
        print_per    = mfrml.smpl(start_, end_ ,altdf_)
        vars         = mfrml.allvar.keys()
        varterms     = [(term.var, int(term.lag) if term.lag else 0)
                        for term in mfrml.allvar[varnavn]['terms'] if term.var and not (term.var ==varnavn and term.lag == '')]
        sterms       = sorted(set(varterms), key=lambda x: varterms.index(x)) # now we have droped dublicate terms and sorted 
        eksperiments = [(vt,t) for vt in sterms  for t in print_per]    # find all the eksperiments to be performed 
        smallalt     = altdf_.loc[:,vars].copy(deep=True)   # for speed 
        smallbase    = basedf_.loc[:,vars].copy(deep=True)  # for speed 
        alldf        = {e: smallalt.copy()   for e in eksperiments}       # make a dataframe for each experiment
        for  e in eksperiments:
              (var_,lag_),per_ = e 
              set_a_value(alldf[e],per_,var_,lag_,get_a_value(smallbase,per_,var_,lag_))
#              alldf[e].loc[e[1]+e[0][1],e[0][0]] = smallbase.loc[e[1]+e[0][1],e[0][0]] # update the variable in each eksperiment
          
        difdf        = {e: smallalt - alldf[e] for e in eksperiments }           # to inspect the updates     
        #allres       = {e : mfrml.xgenr(alldf[e],str(e[1]),str(e[1]),silent= True ) for e in eksperiments} # now evaluate each experiment
        allres       = {e : mfrml.xgenr(alldf[e],e[1],e[1],silent= True ) for e in eksperiments} # now evaluate each experiment
        diffres      = {e: smallalt - allres[e] for e in eksperiments }          # dataframes with the effect of each update 
        res          = {e : diffres[e].loc[e[1],varnavn]  for e in eksperiments}          # we are only interested in the efect on the left hand variable 
    # the resulting dataframe 
        multi        = pd.MultiIndex.from_tuples([e[0] for e in eksperiments],names=['Variable','lag']).drop_duplicates() 
        resdf        = pd.DataFrame(index=multi,columns=print_per)
        for e in eksperiments: 
            resdf.at[e[0],e[1]] = res[e]     
            
    #  a dataframe with some summaries 
        res2df        = pd.DataFrame(index=multi,columns=print_per)
        res2df.loc[('Base','0'),print_per]           = smallbase.loc[print_per,varnavn]
        res2df.loc[('Alternative','0'),print_per]    = smallalt.loc[print_per,varnavn]   
        res2df.loc[('Difference','0'),print_per]     = difendo = smallalt.loc[print_per,varnavn]- smallbase.loc[print_per,varnavn]
        res2df.loc[('Percent   ','0'),print_per]     =  100*(smallalt.loc[print_per,varnavn]/ (0.0000001+smallbase.loc[print_per,varnavn])-1)
        res2df=res2df.dropna()
    #  
        pctendo  = (resdf / (0.000000001+difendo[print_per]) *100).sort_values(print_per[-1],ascending = False)       # each contrinution in pct of total change 
        residual = pctendo.sum() - 100 
        pctendo.at[('Total',0),print_per]     = pctendo.sum() 
        pctendo.at[('Residual',0),print_per]  = residual
        if lprint: 
            print(  'Formula        :',mfrml.allvar[varnavn]['frml'],'\n')
            print(res2df.to_string (float_format=lambda x:'{0:10.6f}'.format(x) ))  
            print('\n Contributions to differende for ',varnavn)
            print(resdf.to_string  (float_format=lambda x:'{0:10.6f}'.format(x) ))
            print('\n Share of contributions to differende for ',varnavn)
            print(pctendo.to_string(float_format=lambda x:'{0:10.0f}%'.format(x) ))
    
        pctendo=pctendo[pctendo.columns].astype(float)    
        return res2df,resdf,pctendo
      
    def treewalk(self,g,navn, level = 0,parent='Start',maxlevel=20,lpre=True):
        ''' Traverse the call tree from name, and returns a generator \n
        to get a list just write: list(treewalk(...)) 
        maxlevel determins the number of generations to back up 
    
        lpre=0 we walk the dependent
        lpre=1 we walk the precednc nodes  
        '''
        if level <=maxlevel:
            if parent != 'Start':
                yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
            for child in (g.predecessors(navn) if lpre else g[navn]):
                yield from self.treewalk(g,child, level + 1,navn, maxlevel,lpre)

    def impact(self,var,ldekomp=False,leq=False,adverse=None,base=None,maxlevel=3,start='',end=''):
        for v in self.treewalk(self.endograph,var.upper(),parent='start',lpre=True,maxlevel=maxlevel):
            if v.lev <= maxlevel:
                if leq:
                    print('---'*v.lev+self.allvar[v.child]['frml'])
                    self.print_eq(v.child.upper(),data=self.lastdf,start='2015Q4',slut='2018Q2')
                else:
                    print('---'*v.lev+v.child)                    
                if ldekomp :
                    x=self.dekomp(v.child,lprint=1,start=start,end=end)
                
    
     
    def drawendo(self,**kwargs):
       '''draws a graph of of the whole model''' 
       alllinks = (node(0,n[1],n[0]) for n in self.endograph.edges())
       return self.todot2(alllinks,**kwargs) 

    def drawmodel(self,lag=True,**kwargs):
        '''draws a graph of of the whole model''' 
        graph = self.totgraph if lag else self.totgraph_nolag
        alllinks = (node(0,n[1],n[0]) for n in graph.edges())
        return self.todot2(alllinks,**kwargs) 
   
    def draw(self,navn,down=7,up=7,lag=True,endo=False,**kwargs):
       '''draws a graph of dependensies of navn up to maxlevel
       
       :lag: show the complete graph including lagged variables else only variables. 
       :endo: Show only the graph for current endogenous variables 
       :down: level downstream
       :up: level upstream 
       
       
       '''
       graph = self.totgraph if lag else self.totgraph_nolag
       graph = self.endograph if endo else graph    
       uplinks   = self.treewalk(graph,navn.upper(),maxlevel=up,lpre=True)       
       downlinks = (node(level , navn, parent ) for level,parent,navn in 
                    self.treewalk(graph,navn.upper(),maxlevel=down,lpre=False))
       alllinks  = chain(uplinks,downlinks)
       return self.todot2(alllinks,navn=navn.upper(),down=down,up=up,**kwargs) 
   
    
    def trans(self,ind,root,transdic=None,debug=False):
        ''' as there are many variable starting with SHOCK, the can renamed to save nodes'''
        if debug:    print('>',ind)
        ud = ind
        if ind == root or transdic is None:
            pass 
        else: 
            for pat,to in transdic.items():
                if debug:  print('trans ',pat,ind)
                if bool(re.match(pat.upper(),ind)):
                    if debug: print(f'trans match {ind} with {pat}')
                    return to.upper()
            if debug: print('trans',ind,ud) 
        return ud    
         
    def color(self,v,navn=''):
        if navn == v:
            out = 'Turquoise'
            return out 
        if v in self.endogene:  
                out = 'steelblue1' 
        elif v in self.exogene:    
                out = 'yellow'    
        elif '(' in v:
                namepart=v.split('(')[0]
                out = 'springgreen' if namepart in self.endogene else 'olivedrab1'
        else:
            out='red'
        return out

    def upwalk(self,g,navn, level = 0,parent='Start',up=20,select=False,lpre=True):
        ''' Traverse the call tree from name, and returns a generator \n
        to get a list just write: list(upwalk(...)) 
        up determins the number of generations to back up 
    
        '''
        if select: 
            if level <=up:
                if parent != 'Start':
                        if (   g[navn][parent]['att'] != 0.0).any(axis=1).any() :
                            yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
                for child in (g.predecessors(navn) if lpre else g[navn]) :
                         try:
                             if (   g[child][navn]['att'] != 0.0).any(axis=1).any() :
                                yield from self.upwalk(g,child, level + 1,navn, up,select)
                         except:
                             pass
        else:
            if level <=up:
                if parent != 'Start':
                    yield  node(level , parent, navn)   # return level=0 in order to prune dublicates 
                for child in (g.predecessors(navn) if lpre else g[navn]):
                    yield from self.upwalk(g,child, level + 1,navn, up,select,lpre)


    def explain(self,var,up=1,start='',end='',select=False,showatt=True,lag=True,debug=0,**kwargs):
        ''' Walks a tree to explain the difference between basedf and lastdf
        
        Parameters:
        :var:  the variable we are looking at
        :up:  how far up the tree will we climb
        :select: Only show the nodes which contributes 
        :showatt: Show the explanation in pct 
        :lag:  If true, show all lags, else aggregate lags for each variable. 
        :HR: if true make horisontal graph
        :title: Title 
        :saveas: Filename 
        :pdf: open the pdf file
        :svg: display the svg file
        :browser: if true open the svg file in browser 
            
            
        '''    
        if up > 0:
            with ttimer('Get totgraph',debug) as t: 
                startgraph = self.totgraph  # if lag else self.totgraph_nolag
            edgelist = list({v for v in self.upwalk(startgraph ,var.upper(),up=up)})
            nodelist = list({v.child for v in edgelist})+[var] # remember the starting node 
            nodestodekomp = list({n.split('(')[0] for n in nodelist if n.split('(')[0] in self.endogene})
    #        print(nodelist)
    #        print(nodestodekomp)
            with ttimer('Dekomp',debug) as t: 
                pctdic2 = {n : self.get_att_pct(n,lag=lag,start=start,end=end) for n in nodestodekomp }
            edges = {(r,n):{'att':df.loc[[r],:]} for n,df in pctdic2.items() for r in df.index}
            self.localgraph  = nx.DiGraph()
            self.localgraph.add_edges_from([(v.child,v.parent) for v in edgelist])
            nx.set_edge_attributes(self.localgraph,edges)
            self.newgraph = nx.DiGraph()
            for v in self.upwalk(self.localgraph,var.upper(),up=up,select=select):
    #                print(f'{"-"*v.lev} {v.child} {v.parent} \n',self.localgraph[v.child][v.parent].get('att','**'))
    #                print(f'{"-"*v.lev} {v.child} {v.parent} \n')
                    self.newgraph.add_edge(v.child,v.parent,att=self.localgraph[v.child][v.parent].get('att',None))
            nodeatt = {n:{'att': i} for n,i in pctdic2.items()}        
            nx.set_node_attributes(self.newgraph,nodeatt)
            nodevalues = {n:{'values':self.get_values(n)} for n in self.newgraph.nodes}
            nx.set_node_attributes(self.newgraph,nodevalues)
        else:     
            self.newgraph = nx.DiGraph([(var,var)])
            nx.set_node_attributes(self.newgraph,{var:{'values':self.get_values(var)}})
            nx.set_node_attributes(self.newgraph,{var:{'att':self.get_att_pct(var,lag=lag,start=start,end=end)}})
        self.gdraw(self.newgraph,navn=var,showatt=showatt,**kwargs)
        return self.newgraph



    
    def todot(self,g,navn='',browser=False,**kwargs):
        ''' makes a drawing of subtree originating from navn
        all is the edges
        attributex can be shown
        
        :sink: variale to use as sink 
        :svg: Display the svg image 
''' 
        size=kwargs.get('size',(6,6))
    
        alledges = (node(0,n[1],n[0]) for n in g.edges())

        if 'transdic' in kwargs:
            alllinks = (node(x.lev,self.trans(x.parent,navn,kwargs['transdic']),self.trans(x.child,navn,kwargs['transdic']))  for x in alledges)
        elif hasattr(self, 'transdic'):
            alllinks = (node(x.lev,self.trans(x.parent,navn,self.transdic),self.trans(x.child,navn,self.transdic))  for x in alledges)
        else:
            alllinks = alledges 
            
     
        ibh  = {node(0,x.parent,x.child) for x in alllinks}  # To weed out multible  links 
        
        if kwargs.get('showatt',False):
            att_dic = nx.get_node_attributes(g,'att')
            values_dic = nx.get_node_attributes(g,'values')
            showatt=True
        else:
            showatt=False
    #
        dec = kwargs.get('dec',0)
        nodelist = {n for nodes in ibh for n in (nodes.parent,nodes.child)}
        
        def dftotable(df,dec=0):
             xx = '\n'.join([f"<TR><TD ALIGN='LEFT'>{row[0]}</TD>"+
                             ''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() 
                                    for b in row[1:]])+'</TR>' for row in df.itertuples()])
             return xx               
        def makenode(v):
#            tip= f'{pt.split_frml(self.allvar[v]["frml"])[3][:-1]}' if v in self.endogene else f'{v}'   
            tip = v 
            if showatt:
                dfval = values_dic[v]
                dflen = len(dfval.columns)
                lper = "<TR><TD ALIGN='LEFT'>Per</TD>"+''.join([ '<TD>'+(f'{p}'.strip()+'</TD>').strip() for p in dfval.columns])+'</TR>'
                hval = f"<TR><TD COLSPAN = '{dflen+1}'>{tip}</TD></TR>" 
                lval   = dftotable(dfval,dec)
                try:                    
                    latt =  f"<TR><TD COLSPAN = '{dflen+1}'> % Explained by</TD></TR>{dftotable(att_dic[v],dec)}" if len(att_dic[v]) else ''
                except: 
                    latt = ''
                    
                linesout=hval+lper+lval+latt   
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  '+ (
                f" label=<<TABLE BORDER='1' CELLBORDER = '1'  > {linesout} </TABLE>> ]")
                pass
            else:
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue style=filled  '+ (
                f" label=<<TABLE BORDER='0' CELLBORDER = '0' {tip} > <TR><TD>{v}</TD></TR> </TABLE>> ]")
            return out    
        
        pre   = 'Digraph TD { rankdir ="HR" \n' if kwargs.get('HR',False) else 'Digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n '+ '\n'.join([makenode(v) for v in nodelist])+' \n} \n'
        
        def getpw(v):
            try:
               return max(0.5,min(5.,abs(g[v.child][v.parent]['att'].iloc[0,-1])/20.))
            except:
               return 0.5
           
        if showatt:
            pw = [getpw(v) for v in ibh]
        else: 
            pw= [1 for v in ibh]
            
        links = '\n'.join([f'"{v.child}" -> "{v.parent}" [penwidth={p}]'  for v,p in zip(ibh,pw)])
    
                    
        psink = '\n{ rank = sink; "'+kwargs['sink'].upper()+'"  ; }' if  kwargs.get('sink',False) else ''
        psource = '\n{ rank = source; "'+kwargs['source'].upper()+'"  ; }' if  kwargs.get('source',False) else ''
        fname = kwargs.get('saveas',f'{navn} explained' if navn else "A_model_graph")  
                  
        ptitle = '\n label = "'+kwargs.get('title',fname)+'";'
        post  = '\n}' 

        out   = pre+nodes+links+psink+psource+ptitle+post 
        tpath=os.path.join(os.getcwd(),'graph')
        if not os.path.isdir(tpath):
            try:
                os.mkdir(tpath)
            except: 
                print("ModelFlow: Can't create folder for graphs")
                return 
    #    filename = os.path.join(r'graph',navn+'.gv')
        filename = os.path.join(tpath,fname+'.gv')
        pngname  = '"'+os.path.join(tpath,fname+'.png')+'"'
        svgname  = '"'+os.path.join(tpath,fname+'.svg')+'"'
        pdfname  = '"'+os.path.join(tpath,fname+'.pdf')+'"'
        epsname  = '"'+os.path.join(tpath,fname+'.eps')+'"'

        with open(filename,'w') as f:
            f.write(out)
#        run('dot -Tsvg  -Gsize=19,19\! -o'+svgname+' "'+filename+'"',shell=True) # creates the drawing  
#        run('dot -Tpng  -Gsize=9,9\! -o'+pngname+' "'+filename+'"',shell=True) # creates the drawing  
#        run('dot -Tpdf  -Gsize=9,9\! -o'+pdfname+' "'+filename+'"',shell=True) # creates the drawing  
        run(f'dot -Tsvg  -Gsize={size[0]},{size[1]}\! -o{svgname} "{filename}"',shell=True) # creates the drawing  
        run(f'dot -Tpng  -Gsize={size[0]},{size[1]}\! -o{pngname} "{filename}"',shell=True) # creates the drawing  
        run(f'dot -Tpdf  -Gsize={size[0]},{size[1]}\! -o{pdfname} "{filename}"',shell=True) # creates the drawing  

#        run('dot -Teps  -Gsize=9,9\! -o'+epsname+' "'+filename+'"',shell=True) # creates the drawing  
        if 'svg' in kwargs:
            display(SVG(filename=svgname[1:-1]))
        else:            
            display(Image(filename=pngname[1:-1]))
            
        if kwargs.get('pdf',False)     : os.system(pdfname)
        if kwargs.get('browser',False) : wb.open(svgname,new=2)
        
        return 
                    
    def gdraw(self,g,**kwargs):
        '''draws a graph of of the whole model''' 
        out=self.todot(g,**kwargs)
        return out


    def todot2(self,alledges,navn='',browser=False,**kwargs):
        ''' makes a drawing of all edges in list alledges
        all is the edges
        
        
        :all: show values for .dfbase and .dflaste
        :last: show the values for .dflast 
        :sink: variale to use as sink 
        :source: variale to use as ssource 
        :svg: Display the svg image in browser
        :pdf: display the pdf result in acrobat reader 
        :saveas: Save the drawing as name 
        :size: figure size default (6,6)
        :warnings: warnings displayed in command console, default =False 
        :invisible: set of invisible nodes 
        :labels: dict of labels for edges 
        :transdic: dict of translations for consolidation of nodes {'SHOCK[_A-Z]*__J':'SHOCK__J','DEV__[_A-Z]*':'DEV'}
        :dec: decimal places in numbers
        :HR: horisontal orientation default = False 
      
        
''' 
        
        
        invisible = kwargs.get('invisible',set())
        labelsdic = kwargs.get('labels',{})
        size=kwargs.get('size',(6,6))
        
        class defsub(dict):
            '''A subclass of dict.
            if a *defsub* is indexed by a nonexisting keyword it just return the keyword '''
            
            def __missing__(self, key):
                return key 
        #%
        labels = defsub(labelsdic)
        
        def stylefunk(n1=None,n2=None,invisible=set()):
            if n1 in invisible or n2 in invisible:
                if n2:
                    return 'style = invisible arrowhead=none '
                else: 
                    return 'style = invisible '

            else:
#                return ''
                return 'style = filled'
        def stylefunkhtml(n1=None,invisible=set()):
#            return ''
            if n1 in invisible:
                return 'style = "invisible" '
            else:
                return 'style = "filled"'
            
        if 'transdic' in kwargs:
            alllinks = (node(x.lev,self.trans(x.parent,navn,kwargs['transdic']),self.trans(x.child,navn,kwargs['transdic']))  for x in alledges)
        elif hasattr(self, 'transdic'):
            alllinks = (node(x.lev,self.trans(x.parent,navn,self.transdic)     ,self.trans(x.child,navn,self.transdic))  for x in alledges)
        else:
            alllinks = alledges 
            
     
        ibh  = {node(0,x.parent,x.child) for x in alllinks}  # To weed out multible  links 
    #
        nodelist = {n for nodes in ibh for n in (nodes.parent,nodes.child)}
#        print(nodelist)
        def makenode(v):
            if kwargs.get('last',False) or kwargs.get('all',False):
                try:
                    t = pt.udtryk_parse(v,funks=[])
                    var=t[0].var
                    lag=int(t[0].lag) if t[0].lag else 0
                    dec = kwargs.get('dec',3)
                    bvalues = [float(get_a_value(self.basedf,per,var,lag)) for per in self.current_per] if kwargs.get('all',False)  else 0
                    lvalues = [float(get_a_value(self.lastdf,per,var,lag)) for per in self.current_per] if kwargs.get('last',False) or kwargs.get('all',False)  else 0
                    dvalues = [float(get_a_value(self.lastdf,per,var,lag)-get_a_value(self.basedf,per,var,lag)) for per in self.current_per] if kwargs.get('all',False) else 0 
                    per   = "<TR><TD ALIGN='LEFT'>Per</TD>"+''.join([ '<TD>'+(f'{p}'.strip()+'</TD>').strip() for p in self.current_per])+'</TR>'  
                    base   = "<TR><TD ALIGN='LEFT'>Base</TD>"+''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in bvalues])+'</TR>' if kwargs.get('all',False) else ''    
                    last   = "<TR><TD ALIGN='LEFT'>Last</TD>"+''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in lvalues])+'</TR>' if kwargs.get('last',False) or kwargs.get('all',False) else ''    
                    dif    = "<TR><TD ALIGN='LEFT'>Diff</TD>"+''.join([ "<TD ALIGN='RIGHT'>"+(f'{b:{25},.{dec}f}'.strip()+'</TD>').strip() for b in dvalues])+'</TR>' if kwargs.get('all',False) else ''    
#                    tip= f' tooltip="{self.allvar[var]["frml"]}"' if self.allvar[var]['endo'] else f' tooltip = "{v}" '  
                    out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} '+ (
                    f" label=<<TABLE BORDER='1' CELLBORDER = '1' {stylefunkhtml(var,invisible=invisible)} > <TR><TD COLSPAN ='{len(lvalues)+1}'>{labels[v]}</TD></TR>{per} {base}{last}{dif} </TABLE>> ]")
                    pass 

                except:
                   out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue {stylefunk(var,invisible=invisible)} '+ (
                    f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(var,invisible=invisible)} > <TR><TD>{labels[v]}</TD></TR> <TR><TD> Condensed</TD></TR></TABLE>> ]")
            else:
                out = f'"{v}" [shape=box fillcolor= {self.color(v,navn)} margin=0.025 fontcolor=blue {stylefunk(v,invisible=invisible)} '+ (
                f" label=<<TABLE BORDER='0' CELLBORDER = '0' {stylefunkhtml(v,invisible=invisible)}  > <TR><TD>{labels[v]}</TD></TR> </TABLE>> ]")
            return out    
        
        pre   = 'Digraph TD {rankdir ="HR" \n' if kwargs.get('HR',True) else 'Digraph TD { rankdir ="LR" \n'
        nodes = '{node  [margin=0.025 fontcolor=blue style=filled ] \n '+ '\n'.join([makenode(v) for v in nodelist])+' \n} \n'
 
        links = '\n'.join(['"'+v.child+'" -> "'+v.parent+'"' + f'[ {stylefunk(v.child,v.parent,invisible=invisible)}   ]'    for v in ibh ])
    
        psink   = '\n{ rank = sink; "'  +kwargs['sink']  +'"  ; }' if  kwargs.get('sink',False) else ''
        psource = '\n{ rank = source; "'+kwargs['source']+'"  ; }' if  kwargs.get('source',False) else ''
        clusterout=''
        if kwargs.get('cluster',False): # expect a dict with clustername as key and a list of nodes as content 
            clusterdic = kwargs.get('cluster',False)
            
            for i,(c,cl) in enumerate(clusterdic.items()):
                varincluster = ' '.join([f'"{v.upper()}"' for v in cl])
                clusterout = clusterout + f'\n subgraph cluster{i} {{ {varincluster} ; label = "{c}" ; color=lightblue ; style = filled ;fontcolor = yellow}}'     
            
        fname = kwargs.get('saveas',navn if navn else "A_model_graph")  
        ptitle = '\n label = "'+kwargs.get('title',fname)+'";'
        post  = '\n}' 

        out   = pre+nodes+links+psink+psource+clusterout+ptitle+post 
        tpath=os.path.join(os.getcwd(),'graph')
        if not os.path.isdir(tpath):
            try:
                os.mkdir(tpath)
            except: 
                print("ModelFlow: Can't create folder for graphs")
                return 
    #    filename = os.path.join(r'graph',navn+'.gv')
        filename = os.path.join(tpath,fname+'.gv')
        pngname  = '"'+os.path.join(tpath,fname+'.png')+'"'
        svgname  = '"'+os.path.join(tpath,fname+'.svg')+'"'
        pdfname  = '"'+os.path.join(tpath,fname+'.pdf')+'"'
        epsname  = '"'+os.path.join(tpath,fname+'.eps')+'"'

        with open(filename,'w') as f:
            f.write(out)
        warnings = "" if kwargs.get("warnings",False) else "-q"    
#        run('dot -Tsvg  -Gsize=9,9\! -o'+svgname+' "'+filename+'"',shell=True) # creates the drawing  
        run(f'dot -Tsvg  -Gsize={size[0]},{size[1]}\! -o{svgname} "{filename}"   {warnings} ',shell=True) # creates the drawing  
        run(f'dot -Tpng  -Gsize={size[0]},{size[1]}\! -o{pngname} "{filename}"  {warnings} ',shell=True) # creates the drawing  
        run(f'dot -Tpdf  -Gsize={size[0]},{size[1]}\! -o{pdfname} "{filename}"  {warnings} ',shell=True) # creates the drawing  
#        run('dot -Tpdf  -Gsize=9,9\! -o'+pdfname+' "'+filename+'"',shell=True) # creates the drawing  
#        run('dot -Teps  -Gsize=9,9\! -o'+epsname+' "'+filename+'"',shell=True) # creates the drawing  

        if 'svg' in kwargs:
            display(SVG(filename=svgname[1:-1]))
        else:            
            display(Image(filename=pngname[1:-1]))
        if browser: wb.open(svgname,new=2)
        if kwargs.get('pdf',False)     : os.system(pdfname)

       # run('%windir%\system32\mspaint.exe '+ pngname,shell=True) # display the drawing 
        return 
     
    
    def vis(self,*args,**kwargs):
        ''' Visualize the data of this model instance 
        if the user has another vis class she can place it in _vis, then that will be used'''
        if not hasattr(self,'_vis'):
           self._vis = mv.vis
        return self._vis(self,*args,**kwargs)

    def varvis(self,*args,**kwargs):
        return mv.varvis(self,*args,**kwargs)

        
    def compvis(self,*args,**kwargs):
        return mv.compvis(self,*args,**kwargs)
    

    def todynare(self,paravars=[],paravalues=[]):
        ''' This is a function which converts a Pyfs model instance to Dynare .mod format
        ''' 
        def totext(t):
            if t.op:
                return  ';\n' if ( t.op == '$' ) else t.op.lower()
            elif t.number:
                return  t.number
            elif t.var:
                return t.var+(('('+t.lag+')') if t.lag else '') 
        
        content = ('//  Dropped '+v +'\n' if self.allvar[v]['dropfrml']
               else ''.join( (totext(t) for t in self.allvar[v]['terms']))    
                   for v in self.solveorder )

        paraout = ('parameters \n  ' + '\n  '.join(v for v in sorted(paravars)) + ';\n\n' +  
                      ';\n'.join(v for v in paravalues)+';\n')
#        print(paraout)
        out = (  '\n'.join(['@#define '+k+' = [' + ' , '.join
                 (['"'+d+'"' for d in  kdic])+']' for l,dic in self.lister.items() for k,kdic in dic.items()]) + '\n' +
                'var    \n  ' + '\n  '.join((v for v in sorted(self.endogene)))+';\n'+
                'varexo \n  ' + '\n  '.join(sorted(self.exogene-set(paravars))) + ';\n'+ 
                paraout + 
                'model; \n  ' + '  '.join(content).replace('**','^') + ' \n end; \n' )
        
        return out
    
    def itershow(self,per=''):
            ''' dunmps iterations ''' 
            model = self
        #try:
            per_ = model.current_per[-1] if per == '' else per  
            indf = model.dumpdf.query('per == @per_')     
            out= indf.query('per == @per_').set_index('iteration',drop=True).drop('per',axis=1).copy()
            vars = [v for  v in out.columns if v in model.endogene]  
            number = out.shape[1] 
            #print(vars)
            #print(out.head())
            axes=out[vars].plot(kind='line',subplots=True,layout=(number,1),figsize = (10, number*3),
                 use_index=True,title=f'Iterations in {per_} ',sharey=0)
            fig = axes.flatten()[0].get_figure()
            fig.tight_layout()
            fig.subplots_adjust(top=0.97)
            return fig
        #except:
            print('No iteration dump' )

       
def create_strong_network(g,name='Network',typeout=False,show=False):
    ''' create a solveorder and   blockordering of af graph 
    uses networkx to find the core of the model
    ''' 
    strong_condensed = nx.condensation(g)
    strong_topo      = list(nx.topological_sort(strong_condensed))
    solveorder       = [v   for s in strong_topo for v in strong_condensed.node[s]['members']]
    if typeout: 
        block = [[v for v in strong_condensed.node[s]['members']] for s in strong_topo]
        type  = [('Simultaneous'+str(i) if len(l)  !=1 else 'Recursiv ',l) for i,l in enumerate(block)] # count the simultaneous blocks so they do not get lumped together 
# we want to lump recursive equations in sequense together 
        strongblock = [[i for l in list(item) for i in l[1]  ] for key,item in groupby(type,lambda x: x[0])]
        strongtype  = [list(item)[0][0][:-1]                   for key,item in groupby(type,lambda x: x[0])]
        if show:
            print('Blockstructure of:',name)
            print(*Counter(strongtype).most_common())
            for i,(type,block) in enumerate(zip(strongtype,strongblock)):
                print('{} {:<15} '.format(i,type),block)
        return solveorder,strongblock,strongtype   
    else:  
        return solveorder    

#  Functions used in calculating 
       

# wrapper 
    
def create_model(navn, hist=0, name='',new=True,finished=False,xmodel=model,straight=False,funks=[]):
    '''Creates either a model instance or a model and a historic model from formulars. \n
    The formulars can be in a string or in af file withe the extension .txt 
    
    if:
    
    :navn: The model as text or as a file with extension .txt
    :name: Name of the model     
    :new: If True, ! used for comments, else () can also be used. False should be avoided, only for old PCIM models.  
    :hist: If True, a model with calculations of historic value is also created
    :xmodel: The model class used for creating model the model instance. Can be used to create models with model subclasses
    :finished: If True, the model exploder is not used.
    :straight: If True, the formula sequence in the model will be used.
    :funks: A list of user defined funktions used in the model 
    
    
      
             
    '''
    shortname = navn[0:-4] if '.txt' in navn else name if name else 'indtastet_udtryk'
    modeltext = open(navn).read().upper() if '.txt' in navn else navn.upper()
    modeltext= modeltext if new else re.sub(r'\(\)','!',modeltext)
    udrullet = modeltext if finished else mp.explode(modeltext,funks=funks)
    pt.check_syntax_model(udrullet)
    
    if hist:
        hmodel = mp.find_hist_model(udrullet)
        pt.check_syntax_model(hmodel)
        mp.modelprint(hmodel,'Historic calculations ' +shortname,udfil=shortname + '_hist.fru'   ,short=False)
        return xmodel(udrullet, shortname,straight=straight,funks=funks), xmodel(hmodel, shortname + '_hist',straight=straight,funks=funks)
    else:
        return xmodel(udrullet, shortname,straight=straight,funks=funks)
#%%
def get_a_value(df,per,var,lag=0):
    ''' returns a value for row=p+lag, column = var 
    
    to take care of non additive row index'''
    
    return df.iat[df.index.get_loc(per)+lag,df.columns.get_loc(var)]

def set_a_value(df,per,var,lag=0,value=np.nan):
    ''' Sets a value for row=p+lag, column = var 
    
    to take care of non additive row index'''
    
    df.iat[df.index.get_loc(per)+lag,df.columns.get_loc(var)]=value
#%%                   
def insertModelVar(dataframe, model=None):
    """Inserts all variables from model, not already in the dataframe.
    Model can be a list of models """ 
    if isinstance(model,list):
        imodel=model
    else:
        imodel = [model]

    myList=[]
    for item in imodel: 
        myList.extend(item.allvar.keys())
    manglervars = list(set(myList)-set(dataframe.columns))
    if len(manglervars):
        extradf = pd.DataFrame(0.0,index=dataframe.index,columns=manglervars).astype('float64')
        data = pd.concat([dataframe,extradf],axis=1)        
        return data
    else:
        return dataframe
#%%
def lineout(vek,pre='',w=20 ,d=0,pw=20, endline='\n'):
    ''' Utility to return formated string of vector '''
    fvek=[float(v) for v in vek]
    return f'{pre:<{pw}} '+ " ".join([f'{f:{w},.{d}f}' for f in fvek])+endline
#print(lineout([1,2,300000000],'Forspalte',pw=10))

def dfout(df,pre='',w=2 ,d=0,pw=0):
    pw2= pw if pw else max( [len(str(i)) for i in df.index])
    return ''.join([lineout(row,index,w,d,pw2) for index,row in df.iterrows()])
        
#print(dfout(df.iloc[:,:4],w=10,d=2,pw=10))
#%%
def upddfold(base,upd):
    ''' takes two dataframes. The values from upd is inserted into base ''' 
    rows = [(i,base.index.get_loc(ind)) for (i,ind) in  enumerate(upd.index)]
            
    locb = {v:i for i,v in enumerate(base.columns)}
    cols = [(i,locb[v]) for i,v in enumerate(upd.columns)]    
    for cu,cb in cols:
        for ru,rb in rows :
            t=upd.get_value(ru,cu,takeable=True)
            base.values[rb,cb] = t
#            base.set_value(rb,cb,t,takeable=True)
    return base

def upddf(base,upd):
    ''' takes two dataframes. The values from upd is inserted into base ''' 
    rows = [(i,base.index.get_loc(ind)) for (i,ind) in  enumerate(upd.index)]
            
    locb = {v:i for i,v in enumerate(base.columns)}
    cols = [(i,locb[v]) for i,v in enumerate(upd.columns)]    
    for cu,cb in cols:
        for ru,rb in rows :
            t=upd.values[ru,cu]
            base.values[rb,cb] = t
#            base.set_value(rb,cb,t,takeable=True)
    return base

def randomdf(df,row=False,col=False,same=False,ran=False,cpre='C',rpre='R'):
    ''' Randomize and rename the rows and columns of a dataframe, keep the values right:
    
    :ran:  If True randomize, if False don't randomize
    :col:  The columns are renamed and randdomized
    :row:  The rows are renamed and randdomized
    :same:  The row and column index are renamed and randdomized the same way
    :cpre:  Column name prefix
    :rpre:  Row name prefix
    
    ''' 
    from random import sample
    from math import log10
    
    dfout=df.copy(deep=True)
    if ran:
        if col or same:
            colnames = dfout.columns
            dec=str(1+int(log10(len(colnames))))
            rancol = sample(list(colnames),k=len(colnames))
            coldic = {c : cpre+('{0:0'+dec+'d}').format(i) for i,c in enumerate(rancol)}
            dfout = dfout.rename(columns=coldic).sort_index(axis=1)
            if same:
               dfout = dfout.rename(index=coldic).sort_index(axis=0) 
        if row and not same: 
            rownames = dfout.index.tolist()
            dec=str(1+int(log10(len(rownames))))           
            ranrow = sample(list(rownames),k=len(rownames))
            rowdic = {c : rpre+('{0:0'+dec+'d}').format(i)  for i,c in enumerate(ranrow)}
            dfout = dfout.rename(index=rowdic).sort_index(axis=0)
    return dfout  
   
@contextmanager
def ttimer(input='test',show=True,short=False):
    """A timer context manager, implemented using a
    generator function. This one will report time even if an exception occurs"""
    start = time.time()
    if show and not short: print(f'{input} started at : {time.strftime("%H:%M:%S"):>{15}} ')
    try:
        yield
    finally:
        if show:  
            end = time.time()
            seconds = (end - start)
            minutes = seconds/60. 
            if minutes < 2.:
                afterdec='1' if seconds >= 10 else ('3' if seconds >= 1 else '10')
                print(f'{input} took       : {seconds:>{15},.{afterdec}f} Seconds')
            else:
                afterdec='1' if minutes >= 10 else '4'
                print(f'{input} took       : {minutes:>{15},.{afterdec}f} Minutes')
                
if __name__ == '__main__' :
    if 0:
#%% Test model 
        ftest = ''' FRMl <>  y = c + i + x $ 
        FRMl <>  yd = 0.6 * y $
        frml <>  c=0.8*yd $
        FRMl <>  i = ii+iy $ 
        FRMl <>  ii = x+z $
        FRMl <>  x = 2 $ 
        FRML <>  dogplace = y *4 $'''
        mtest = model(ftest)
        mtest.drawall()
        mtest.drawendo()
        mtest.drawpre('Y')
        mtest.todot('Y')
        g = mtest.endograph
#        print(mtest.strongblock)
#        print(mtest.strongtype)
#        (list(mtest.treewalk(mtest.allgraph,'Y',lpre=1)))
#%%       
    if 0:
#%%        
        if 'base0' not in locals():
            with open(r"models\mtotal.fru", "r") as text_file:
                ftotal = text_file.read()
            base0   = pd.read_pickle(r'data\base0.pc')    
            adve0   = pd.read_pickle(r'data\adve0.pc')     
            base    = pd.read_pickle(r'data\base.pc')    
            adverse = pd.read_pickle(r'data\adverse.pc') 
    #%%  Transpile the model   
        if 1: 
            tdic={'SHOCK[_A-Z]*__J':'SHOCK__J','SHOCK[_A-Z]*__0':'SHOCK__0','DEV__[_A-Z]*':'DEV','SHOCK__[_A-Z]*':'SHOCK'}
            with ttimer('Transpile:'):
                mtotal =  model(ftotal,straight=False)
            b = mtotal(base,'2016q1','2018q4')    
            a = mtotal(adverse,'2016q1','2018q4',samedata=True)    
            assert 1==2
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=1,lag=True)
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=0,down=5)
            mtotal.draw('rcet1__DECOM'.upper(),up=1,down=7,browser=False)
            mtotal.draw('rcet1__DECOM'.upper(),uplevel=3,downlevel=7,browser=False)
            mtotal.draw('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',up=11,down=8,sink='IMPACT__DE',browser=True,transdic=tdic)
            mtotal.draw('imp_loss_total_def__DECOM__DE__HH_OTHER_NSME_AIRB',sink='IMPACT__DE',uplevel=12,downlevel=11,browser=True,transdic=tdic)
#            with ttimer('Second calculation new translation :'):
#                adverse2  = mtotal(adve0   ,'2016q1','2018Q4',samedata=True,silent=True)
            mtotal.impact('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',leq=1)
#            mtotal.impact('REA_NON_DEF__DECOM__DE__HH_OTHER_NSME_AIRB',ldekomp=1)
#%%            
        if 0:
#%%     Get a plain model    
            with ttimer('MAKE model and base+adverse'):
                mtotal =  model(ftotal,straight=True)
                base2     = mtotal(base0   ,'2016q1','2018Q4',samedata=True,silent=True)
                adverse2  = mtotal(adve0   ,'2016q1','2018Q4',samedata=True,silent=True,sim=True,antal=3)
            assert (base2 == base).all().all()
            assert (adverse2 == adverse).all().all()
#%%
        if 0:
            a=mtotal.vis('PD__*__CY').plot(ppos=-2)
            compvis(mtotal,'PD__FF_HH_H*').box()

    #%%
    if 1:
#%% a simpel model         
        numberlines = 3
        df = pd.DataFrame({'A':[1.,2.,0.0002,4004.00003] , 'B':[10.,20.,30.,40.] })
        df2 = pd.DataFrame({'A':[1.,2.,0.0002,4010.00003] , 'B':[10.,20.,30.,50.] })
        m2test=model('frml <z> d = log(11) $ \n frml xx yy = a0 + yy(-1)+yy(-2) + +horse**3 + horse(-1)+horse(-2) $'+''.join(['FRMl <>  a'+str(i)+
                                      '=a +'+str(i) + ' +c $ frml xx d'+str(i) + ' = log(1)+abs(a) $' 
                             for i in range(numberlines)]))
        yy=m2test(df)
        yy=m2test(df2)
        m2test.A1.showdif
        if 0:
            m2test.drawmodel(invisible={'A2'})
            m2test.drawmodel(cluster = {'test1':['horse','A0','A1'],'test2':['YY','D0']},sink='D2')
            m2test.drawendo(last=1,size=(2,2))
            m2test.drawmodel(all=1,browser=0,lag=0,dec=4,HR=0,title='test2',invisible={'A2'},labels={'A0':'This is a test'})
            print(m2test.equations.replace('$','\n'))
            print(m2test.todynare(paravars=['HORSE','C'],paravalues=['C=33','HORSE=42']))
#%%        
#        m2test.drawmodel(transdic={'D[0-9]':'DDD'},last=1,browser=1)
#        m2test.drawmodel(transdic={'D[0-9]':'DDD'},lag=False)
#        m2test.draw('A0',transdic={'D[0-9]':'DDD'},sink='A',lag=1,all=1)
#%%    
    if 0:
        df = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        ftest = ''' 
        FRMl <>  ii = x+z $
        frml <>  c=0.8*yd $
        FRMl <>  i = ii+iy $ 
        FRMl <>  x = 2 $ 
        FRMl <>  y = c + i + x+ i(-1)$ 
        FRMl <>  yX = 0.6 * y $
        FRML <>  dogplace = y *4 $'''
        m2=model(ftest)
#        m2.drawmodel()
        m2(df)
        m2(df2)
        m2.Y.explain(select=True,showatt=True,HR=False,up=3)
#        g  = m2.ximpact('Y',select=True,showatt=True,lag=True,pdf=0)
        m2.Y.explain(select=0,up=4,pdf=1)
#        m2.Y.dekomp(lprint=1)
        print(m2['I*'])
        
    if 0:

        def f1(e):
            return 42
        
        def f2(c):
            return 103
        df=pd.DataFrame({'A':[1,2],'B':[2,3]})
        mtest =model('frml <> a=b(-1) $',funks =[f1,f2])
        print(mtest.outeval(df))  
        res = mtest(df)
        print(mtest.outsolve())
        res2=mtest(df)
        res3=mtest(df)
    if 0:  
#%%        
        fx = '''
        FRML <I>  x  = y $ 
        FRML <Z> y  = 1-r*x +p*x**2 $ 
        '''
        mx = create_model(fx,straight=True)
        df = pd.DataFrame({'X' : [0.2,0.2] , 'Y' :[0.,0.] , 'R':[1.,0.4] , 'P':[0.,0.4]})
        df
        
        a = mx(df,antal=50,silent=False,alfa=0.8,relconv=0.000001,conv='X',
               debug=False,ldumpvar=0,stats=True,ljit=True)
        b = mx.res(df)
    with ttimer('dddd') as t:
        u=2*2
