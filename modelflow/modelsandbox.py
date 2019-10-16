# -*- coding: utf-8 -*-
"""

This is a module for testing new features of the model class, but in a smaler file. 

Created on Sat Sep 29 06:03:35 2018

@author: hanseni


"""


import sys  
import time



import pandas as pd
from sympy import sympify,Symbol
from collections import defaultdict, namedtuple
import numpy as np 
import scipy as sp
import networkx as nx
import os
from subprocess import run 
import webbrowser as wb
import seaborn as sns 
from IPython.display import SVG, display, Image
import inspect 
from itertools import chain, zip_longest
from numba import jit
import itertools
from collections import namedtuple

import sys  
import time
import re

from modelclass import model, ttimer, insertModelVar
from modelvis import vis

import modelpattern as pt
import modelmanipulation as mp
import modeldiff as md 
from modelmanipulation import split_frml,udtryk_parse
from modelclass import model, ttimer, insertModelVar
from modelinvert import targets_instruments

import modelvis as mv
import modelmf


  
    
class newmodel(model):
    
    def __call__(self, *args, **kwargs ):
            ''' Runs a model. 
            
            Default a straight model is calculated by *xgenr* a simultaneous model is solved by *sim* 
            
            :sim: If False forces a  model to be calculated (not solved) if True force simulation 
            :setbase: If True, place the result in model.basedf 
            :setlast: if False don't place the results in model.lastdf
            
            if the modelproperty previousbase is true, the previous run is used as basedf. 
    
            
            '''
            if hasattr(self,'oldkwargs'):
                newkwargs =  {**self.oldkwargs,**kwargs}
            else:
                newkwargs =  kwargs
                
            self.oldkwargs = newkwargs.copy()
                
            if self.save:
                if self.previousbase and hasattr(self,'lastdf'):
                    self.basedf = self.lastdf.copy(deep=True)
                
            if kwargs.get('sim2',False):
                outdf = self.sim2d(*args, **newkwargs )   
            else: 
                outdf = self.sim1d( *args, **newkwargs) 
    
            if self.save:
                if (not hasattr(self,'basedf')) or kwargs.get('setbase',False) :
                    self.basedf = outdf.copy(deep=True) 
                if kwargs.get('setlast',True)                                  :
                    self.lastdf = outdf.copy(deep=True)
        
            return outdf

  
    @property 
    def showstartnr(self):
        self.findpos()
        variabler=[x for x in sorted(self.allvar.keys())]
        return {v:self.allvar[v]['startnr'] for v in variabler}        
        
       
    def sim2d(self, databank, start='', slut='', silent=0,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=1,conv=[],absconv=0.01,relconv=0.0000000000000001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False,timeon=False,
              fairopt={'fairantal':1},**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
            
        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pro2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_los_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                exec(self.make_los_text2d_jit,globals())  # creates the los function
                self.pro2d_jit,self.solve2d_jit,self.epi2d_jit  = make_los(self.funks,self.errfunk)
            self.pro2d,self.solve2d,self.epi2d = self.pro2d_jit,self.solve2d_jit,self.epi2d_jit
        
        else:
            if newdata or not hasattr(self,'pro2d_nojit'):
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_los_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                exec(self.make_los_text2d_nojit,globals())  # creates the los function
                self.pro2d_nojit,self.solve2d_nojit,self.epi2d_nojit  = make_los(self.funks,self.errfunk)
            self.pro2d,self.solve2d,self.epi2d = self.pro2d_nojit,self.solve2d_nojit,self.epi2d_nojit
            
            
        values = databank.values.copy()  # 
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
 
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pro2d(values, values,  row ,  1.0 )
                for iteration in range(antal):
                    with ttimer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        self.solve2d(values, values, row ,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
                    if iteration > first_test: 
                        itafter=[values[row,c] for c in convplace] 
                        convergence = True
                        for after,before in zip(itafter,itbefore):
    #                        print(before,after)
                            if before > absconv and abs(after-before)/abs(before)  > relconv:
                                convergence = False
                                break 
                        if convergence:
                            break
                        else:
                            itbefore=itafter
                self.epi2d(values, values, row ,  1.0 )

                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 
    


    
    @staticmethod
    def grouper(iterable, n, fillvalue=''):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
    

    def outsolve2dcunk(self,databank, debug=1,chunk=None,ljit=False,type='gauss',cache=False):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        columnsnr=self.get_columnsnr(databank)
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
        print(f'Generating source for {self.name} using ljit = {ljit} ')
        def make_gaussline2(vx,nodamp=False):
            ''' takes a list of terms and translates to a line in a gauss-seidel solver for 
            simultanius models
            the variables            
            
            New version to take hand of several lhs variables. Dampning is not allowed for
            this. But can easely be implemented by makeing a function to multiply tupels
            nodamp is for pre and epilog solutions, which should not be dampened. 
            '''
            termer=self.allvar[vx]['terms']
            assigpos =  self.allvar[vx]['assigpos'] 
            if nodamp:
                ldamp=False
            else:    
                if 'Z' in self.allvar[vx]['frmlname']: # convention for damping equations 
                    assert assigpos == 1 , 'You can not dampen equations with several left hand sides:'+vx
                    endovar=[t.op if t.op else ('values[row,'+str(columnsnr[t.var])+']') for j,t in enumerate(termer) if j <= assigpos-1 ]
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
                    if i > assigpos:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' ) 
                    else:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' ) 
        
            if ldamp: out.append(')') # the last ) in the dampening 
            res = ''.join(out)
            return res+'\n'
            
        def make_resline2(vx,nodamp):
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
                        out.append('outvalues[row'+t.lag+','+str(columnsnr[t.var])+']' )              
                    else:
                        out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' )              
            res = ''.join(out)
            return res+'\n'


        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False,totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True)\n')
            fib1.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')
            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                       else linemake(v,nodamp))   
                           for v in order if len(v)]
            if debug:   
                fib2.append(long+ 'except :\n')
                fib2.append(longer +f'errorfunk(values,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1,content,fib2)),newoverhead+len(content)+len(fib2),neweq

        def makechunkedfunk(name,order,linemake,debug=False,overhead = 0 ,oldeqs = 0,nodamp=False,chunk=None,ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs 
            if chunk == None:
                orderlist = [order]
            else:    
                orderlist = list(self.grouper(order,chunk))
            fib=[]
            fib2=[]
            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
            if ljit:
                fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True,cache=False)\n')
                 
            fib2.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(values,outvalues,row,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  make_resline2 if type == 'res' else make_gaussline2 
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'from numba import jit' + '\n')
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        with ttimer('make model text'):
            if self.use_preorder:
                procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake ,overhead=len(fib1),oldeqs=0,debug=thisdebug, nodamp=True,ljit=ljit,chunk=chunk)
                content,conoverhead,coneqs    = makechunkedfunk('los',self.coreorder,linemake ,overhead=prooverhead,oldeqs=proeqs,debug=thisdebug,ljit=ljit,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder,linemake ,overhead =conoverhead,oldeqs=coneqs,debug=thisdebug,nodamp=True,ljit=ljit,chunk=chunk)
            else:
                procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],linemake ,overhead=len(fib1),oldeqs=0,ljit=ljit,debug=thisdebug,chunk=chunk)
                content,conoverhead,coneqs    =  makechunkedfunk('los',self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
                epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',[],linemake ,ljit=ljit,debug=thisdebug,chunk=chunk,overhead =conoverhead,oldeqs=coneqs)
                
        fib2.append(short + 'return prolog,los,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
    def sim1d(self, databank, start='', slut='', silent=0,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=1,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False, 
              fairopt={'fairantal':1},timeon=0,**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()   
            
        self.findpos()
        databank = insertModelVar(databank,self)   # fill all Missing value with 0.0 
            
        with ttimer('create stuffer and gauss lines ',timeon) as t:        
            if (not hasattr(self,'stuff3')) or  (not self.eqcolumns(self.simcolumns, databank.columns)):
                self.stuff3,self.saveeval3  = self.createstuff3(databank)
                self.simcolumns=databank.columns.copy()

        with ttimer('Create solver function',timeon) as t: 
            if ljit:
                   if not hasattr(self,'solve1d_jit'): 
                       self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, 
                              debug=kwargs.get('debug',1),cache=kwargs.get('cache','False'))
                       exec(self.make_los_text1d,globals())  # creates the los function
                       self.pro1d_jit,self.solve1d_jit,self.epi1d_jit  = make_los(self.funks,self.errfunk)
                   this_pro1d,this_solve1d,this_epi1d = self.pro1d_jit,self.solve1d_jit,self.epi1d_jit
            else:  
                    if not hasattr(self,'solve1d'): 
                        self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                        exec(self.make_los_text1d,globals())  # creates the los function
                        self.pro1d_nojit,self.solve1d_nojit,self.epi1d_nojit  = make_los(self.funks,self.errfunk1d)
 
                    this_pro1d,this_solve1d,this_epi1d = self.pro1d_nojit,self.solve1d_nojit,self.epi1d_nojit                

        values=databank.values.copy()
        self.values_ = values # for use in errdump 
              
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
#        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convplace=[self.allvar[c]['startnr']-self.allvar[c]['maxlead'] for c in convvar]
        convergence = True
 
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else self.vlist(dumpvar)
            dumpplac = [self.allvar[v]['startnr'] -self.allvar[v]['maxlead'] for v in self.dump]

        
        ittotal = 0
        endtimesetup=time.time()

        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                if not silent:
                    print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                self.row_ = row
                with ttimer(f'stuff {self.periode} ',timeon) as t: 
                    a=self.stuff3(values,row,ljit)
#                  
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[a[p] 
                        for p in dumpplac])
    
                itbefore = [a[c] for c in convplace] 
                this_pro1d(a,  1.0 )
                for iteration in range(antal):
                    with ttimer(f'Evaluate {self.periode}/{iteration} ',timeon) as t: 
                        this_solve1d(a,  alfa )
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[a[p]
                              for p in dumpplac])
                    if iteration > first_test: 
                        itafter=[a[c] for c in convplace] 
                        convergence = True
                        for after,before in zip(itafter,itbefore):
    #                        print(before,after)
                            if before > absconv and abs(after-before)/abs(before)  > relconv:
                                convergence = False
                                break 
                        if convergence:
                            break
                        else:
                            itbefore=itafter
                this_epi1d(a ,  1.0 )
                self.saveeval3(values,row,a)
                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            self.dumpdf = self.dumpdf.sort_values(['per','fair','iteration'])
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        del self.values_ # not needed any more 
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 
    
    
    def outsolve1dcunk(self,debug=0,chunk=None,ljit=False,cache='False'):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        self.findpos()
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
        
        
           

        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False,totalchunk=1):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber+1}/{totalchunk}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache})\n')
            fib1.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
                fib1.append(longer+'pass\n')

            newoverhead = len(fib1) + overhead
            content = [longer + ('pass  # '+v +'\n' if self.allvar[v]['dropfrml']
                       else linemake(v,nodamp)+'\n')   
                           for v in order if len(v)]
            if debug:   
                fib2.append(long+ 'except :\n')
                fib2.append(longer +f'errorfunk(a,sys.exc_info()[2].tb_lineno,overhead={newoverhead},overeq={oldeqs})'+'\n')
                fib2.append(longer + 'raise\n')
            fib2.append((long if debug else longer) + 'return \n')
            neweq = oldeqs + len(content)
            return list(chain(fib1,content,fib2)),newoverhead+len(content)+len(fib2),neweq

        def makechunkedfunk(name,order,linemake,debug=False,overhead = 0 ,oldeqs = 0,nodamp=False,chunk=None,ljit=False):
            ''' makes the complete function to evaluate the model. 
            keeps the tab on previous overhead lines and equations, to helt the error function '''

            newoverhead = overhead
            neweqs = oldeqs 
            if chunk == None:
                orderlist = [order]
            else:    
                orderlist = list(self.grouper(order,chunk))
            fib=[]
            fib2=[]
            for i,o in enumerate(orderlist):
                lines,head,eques  = makeafunk(name+str(i),o,linemake,i,debug=debug,overhead=newoverhead,nodamp=nodamp,
                                              ljit=ljit,oldeqs=neweqs,totalchunk=len(orderlist))
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
            if ljit:
                fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+f'@jit("(f8[:],f8)",fastmath=True,cache={cache})\n')
                 
            fib2.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(a,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  self.make_gaussline  
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'import time' + '\n')
        fib1.append(short + 'from numba import jit' + '\n')
       
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        
        if self.use_preorder:
            procontent,prooverhead,proeqs = makechunkedfunk('prolog',self.preorder,linemake , overhead=len(fib1),   oldeqs=0,     ljit=ljit,debug=thisdebug, nodamp=True,chunk=chunk)
            content,conoverhead,coneqs    = makechunkedfunk('los',   self.coreorder,linemake ,overhead=prooverhead, oldeqs=proeqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs    = makechunkedfunk('epilog',self.epiorder, linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,nodamp=True,chunk=chunk)
        else:
            procontent,prooverhead,proeqs  = makechunkedfunk('prolog',[],            linemake ,overhead=len(fib1),  oldeqs=0,      ljit=ljit,debug=thisdebug,chunk=chunk)
            content,conoverhead,coneqs     =  makechunkedfunk('los',  self.solveorder,linemake ,overhead=prooverhead,oldeqs=proeqs, ljit=ljit,debug=thisdebug,chunk=chunk)
            epilog ,epioverhead,epieqs     = makechunkedfunk('epilog',[]             ,linemake ,overhead =conoverhead,oldeqs=coneqs,ljit=ljit,debug=thisdebug,chunk=chunk)
            
        fib2.append(short + 'return prolog,los,epilog\n')
        return ''.join(chain(fib1,procontent,content,epilog,fib2))   
    
  
    def errfunk1d(self,a,linenr,overhead=4,overeq=0):
        ''' Handle errors in sim1d '''
        self.saveeval3(self.values_,self.row_,a)
        self.errfunk(self.values_,linenr,overhead,overeq)
    
    def errfunk(self,values,linenr,overhead=4,overeq=0):
        ''' developement function
        
        to handle run time errors in model calculations'''
        
#        winsound.Beep(500,1000)
        self.errdump = pd.DataFrame(values,columns=self.genrcolumns, index= self.genrindex)
        self.lastdf = self.errdump
        
        print('>> Error in     :',self.name)
        print('>> In           :',self.periode)
        if 0:
            print('>> Linenr       :',linenr)
            print('>> Overhead   :',overhead)
            print('>> over eq   :',overeq)
        varposition = linenr-overhead -1 + overeq
        print('>> varposition   :',varposition)
        
        errvar = self.solveorder[varposition]
        outeq = self.allvar[errvar]['frml']
        print('>> Equation     :',outeq)
        print('A snapshot of the data at the error point is at .errdump ')
        print('Also the .lastdf contains .errdump,  for inspecting ')
        self.print_eq_values(errvar,self.errdump,per=[self.periode])
        if hasattr(self,'dumplist'):
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump

    pass


    def newton1per(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001, nonlin=False ,timeit = False,reset=1,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,ljit=False, 
              fairopt={'fairantal':1},**kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'new2d') :
#           if (not hasattr(self,'solvenew2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_new_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=kwargs.get('debug',1),type='res')
#                exec(self.make_new_text2d,globals())  # creates the los function
#                self.pronew2d,self.solvenew2d,self.epinew2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit

                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff'):
            endovar = self.coreorder if self.use_preorder else self.solveorder
            self.newton_diff = newton_diff(self,forcenum=1,df=databank,
                                           endovar = endovar, ljit=ljit,nchunk=chunk,onlyendocur=True )
        if not hasattr(self,'solver') or reset:
            self.solver = self.newton_diff.get_solve1per(df=databank,periode=[self.current_per[0]])[self.current_per[0]]

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff.endovar]
        
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        for fairiteration in range(fairantal):
            if fairantal >=2:
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
                        for p in dumpplac])
    
                itbefore = [values[row,c] for c in convplace] 
                self.pronew2d(values, values,  row ,  alfa )
                for iteration in range(antal):
                    with ttimer(f'sim per:{self.periode} it:{iteration}',0) as xxtt:
                        before = values[row,newton_col]
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        now   = outvalues[row,newton_col]
                        distance = now-before
                        newton_conv =np.abs(distance).sum()
                        if newton_conv <= 0.000000001 :
        #                    print(f'Iteration {iteration} sum of distances {newton_conv}')
                            break 
                        if iteration != 0 and nonlin and not (iteration % nonlin):
                            with ttimer('Updating solver',timeit) as t3:
                                if not silent :print(f'Updating solver, iteration {iteration}')
                                df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                                self.solver = self.newton_diff.get_solve1per(df=df_now,periode=[self.periode])[self.periode]
                            
                        with ttimer('Update solution',0):
                #            update = self.solveinv(distance)
                            update = self.solver(distance)
                        values[row,newton_col] = before - update
    
                    ittotal += 1
                    
                    if ldumpvar:
                        self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                              for p in dumpplac])
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
                self.epinew2d(values, values, row ,  alfa )
    
                if not silent:
                    if not convergence : 
                        print(f'{self.periode} not converged in {iteration} iterations')
                    else:
                        print(f'{self.periode} Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{self.calculate_freq[-1][1]:>15,}')
            print(f'Total iterations                     :{ittotal:>15,}')
            print(f'Total floating point operations      :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def newtonstack(self, databank, start='', slut='', silent=1,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=20,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=None,nchunk=None,ljit=False,nljit=0, 
              fairopt={'fairantal':1},debug=False,timeit=False,nonlin=False,
              newtonalfa = 1.0 , newtonnodamp=0,forcenum=True,reset = False, **kwargs):
        '''Evaluates this model on a databank from start to slut (means end in Danish). 
        
        First it finds the values in the Dataframe, then creates the evaluater function through the *outeval* function 
        (:func:`modelclass.model.fouteval`) 
        then it evaluates the function and returns the values to a the Dataframe in the databank.
        
        The text for the evaluater function is placed in the model property **make_los_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        ittotal = 0
        diffcount = 0 
        starttimesetup=time.time()
        fairantal = {**fairopt,**kwargs}.get('fairantal',1)
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
            
        if not silent : print ('Will start calculating: ' + self.name)
#        if not samedata or not hasattr(self,'solve2d') :
#           if (not hasattr(self,'solvestack2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#    
#                self.make_losstack_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=debug,type='res')
#                exec(self.make_losstack_text2d,globals())  # creates the los function
#                self.prostack2d,self.solvestack2d,self.epistack2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'pronew2d_jit'):  
                if not silent: print(f'Create compiled solving function for {self.name}')                                 
                self.make_newlos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_jit,globals())  # creates the los function
                self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_jit,self.solvenew2d_jit,self.epinew2d_jit
        
        else:
            if newdata or not hasattr(self,'pronew2d_nojit'):  
                if not silent: print(f'Create solving function for {self.name}')                                 
                self.make_newlos_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_newlos_text2d_nojit,globals())  # creates the los function
                self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit  = make_los(self.funks,self.errfunk)
            self.pronew2d,self.solvenew2d,self.epinew2d = self.pronew2d_nojit,self.solvenew2d_nojit,self.epinew2d_nojit


                
        values = databank.values.copy()
        outvalues = np.empty_like(values)# 
        if not hasattr(self,'newton_diff_stack'):
            self.newton_diff_stack = newton_diff(self,forcenum=1,df=databank,ljit=nljit,nchunk=nchunk)
        if not hasattr(self,'stacksolver'):
            self.getsolver = self.newton_diff_stack.get_solvestacked
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            print(f'Creating new derivatives and new solver')
            self.old_stack_periode = sol_periode.copy()
        elif reset or not all(self.old_stack_periode[[0,-1]] == sol_periode[[0,-1]]) :   
            print(f'Creating new solver')
            diffcount += 1
            self.stacksolver = self.getsolver(databank)
            self.old_stack_periode = sol_periode.copy()

        newton_col = [databank.columns.get_loc(c) for c in self.newton_diff_stack.endovar]
        self.newton_diff_stack.timeit = timeit 
        
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = False
     
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else [v for v in self.vlist(dumpvar) if v in self.endogene]
            dumpplac = [databank.columns.get_loc(v) for v in self.dump]
        
        ittotal = 0
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        self.stackrowindex = np.array([[r]*len(newton_col) for r in self.stackrows]).flatten()
        self.stackcolindex = np.array([newton_col for r in self.stackrows]).flatten()

        
#                if ldumpvar:
#                    self.dumplist.append([fairiteration,self.periode,int(0)]+[values[row,p] 
#                        for p in dumpplac])

#        itbefore = values[self.stackrows,convplace] 
#        self.pro2d(values, values,  row ,  alfa )
        for iteration in range(antal):
            with ttimer(f'\nNewton it:{iteration}',timeit) as xxtt:
                before = values[self.stackrowindex,self.stackcolindex]
                with ttimer('calculate new solution',timeit) as t2:                
                    for row in self.stackrows:
                        self.pronew2d(values, outvalues, row ,  alfa )
                        self.solvenew2d(values, outvalues, row ,  alfa )
                        self.epinew2d(values, outvalues, row ,  alfa )
                        ittotal += 1
                with ttimer('extract new solution',timeit) as t2:                
                    now   = outvalues[self.stackrowindex,self.stackcolindex]
                distance = now-before
                newton_conv =np.abs(distance).sum()
                if not silent:print(f'Iteration  {iteration} Sum of distances {newton_conv:>{15},.{6}f}')
                if newton_conv <= 0.001 :
                    convergence = True 
                    break 
                if iteration != 0 and nonlin and not (iteration % nonlin):
                    with ttimer('Updating solver',timeit) as t3:
                        if not silent :print(f'Updating solver, iteration {iteration}')
                        df_now = pd.DataFrame(values,index=databank.index,columns=databank.columns)
                        self.stacksolver = self.getsolver(df=df_now)
                        diffcount += 1
                        
                    
                with ttimer('Update solution',timeit):
        #            update = self.solveinv(distance)
                    update = self.stacksolver(distance)
                    damp = newtonalfa if iteration <= newtonnodamp else 1.0 
                values[self.stackrowindex,self.stackcolindex] = before - damp * update

                    
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode, int(iteration+1)]+[values[row,p]
                          for p in dumpplac])
    #                if iteration > first_test: 
    #                    itafter=[values[row,c] for c in convplace] 
    #                    convergence = True
    #                    for after,before in zip(itafter,itbefore):
    ##                        print(before,after)
    #                        if before > absconv and abs(after-before)/abs(before)  > relconv:
    #                            convergence = False
    #                            break 
    #                    if convergence:
    #                        break
    #                    else:
    #                        itbefore=itafter
#                self.epistack2d(values, values, row ,  alfa )
    
        if not silent:
            if not convergence : 
                print(f'Not converged in {iteration} iterations')
            else:
                print(f'Solved in {iteration} iterations')
           
        if ldumpvar:
            self.dumpdf= pd.DataFrame(self.dumplist)
            del self.dumplist
            self.dumpdf.columns= ['fair','per','iteration']+self.dump
            if fairantal<=2 : self.dumpdf.drop('fair',axis=1,inplace=True)
            
        outdf =  pd.DataFrame(values,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*ittotal
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.4f}')
            print(f'Total model evaluations              :{ittotal:>15,}')
            print(f'Number of solver update              :{diffcount:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.4f}')
            if self.simtime > 0.0:
                print(f'Floating point operations per second : {numberfloats/self.simtime:>15,.1f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def res2d(self, databank, start='', slut='',debug=False,timeit=False,silent=False,
              chunk=None,ljit=0,alfa=1,stats=0,samedata=False,**kwargs):
        '''calculates the result of a model, no iteration or interaction 
        The text for the evaluater function is placed in the model property **make_res_text** 
        where it can be inspected 
        in case of problems.         
        
        '''
    #    print('new nwwton')
        starttimesetup=time.time()
        sol_periode = self.smpl(start,slut,databank)
        if self.maxlag and not (self.current_per[0]+self.maxlag) in databank.index :
            print('***** Warning: You are solving the model before all lags are avaiable')
            print('Maxlag:',self.maxlag,'First solveperiod:',self.current_per[0],'First dataframe index',databank.index[0])
            sys.exit()     
        if self.maxlead and not (self.current_per[-1]-self.maxlead) in databank.index :
            print('***** Warning: You are solving the model before all leads are avaiable')
            print('Maxlag:',self.maxlead,'Last solveperiod:',self.current_per[0],'Last dataframe index',databank.index[1])
            sys.exit()     
        if not silent : print ('Will start calculating: ' + self.name)
        databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
#        if not samedata or not hasattr(self,'solve2d') :
#           if (not hasattr(self,'solve2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
#                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
#                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
#                with ttimer('make model:'):
#                    self.make_res_text2d =  self.outsolve2dcunk(databank,chunk=chunk,
#                      ljit=ljit, debug=debug,type='res')
#                exec(self.make_res_text2d,globals())  # creates the los function
#                self.pro2d,self.solve2d,self.epi2d  = make_los(self.funks,self.errfunk)

        if not self.eqcolumns(self.genrcolumns,databank.columns):
            databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
            for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 
            newdata = True 
        else:
            newdata = False
            
        if ljit:
            if newdata or not hasattr(self,'prores2d_jit'):  
                if not silent: print(f'Create compiled res function for {self.name}')                                 
                self.make_reslos_text2d_jit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_reslos_text2d_jit,globals())  # creates the los function
                self.prores2d_jit,self.solveres2d_jit,self.epires2d_jit  = make_los(self.funks,self.errfunk)
            self.prores2d,self.solveres2d,self.epires2d = self.prores2d_jit,self.solveres2d_jit,self.epires2d_jit
        
        else:
            if newdata or not hasattr(self,'prores2d_nojit'):  
                if not silent: print(f'Create res function for {self.name}')                                 
                self.make_res_text2d_nojit =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1),type='res')
                exec(self.make_res_text2d_nojit,globals())  # creates the los function
                self.prores2d_nojit,self.solveres2d_nojit,self.epires2d_nojit  = make_los(self.funks,self.errfunk)
            self.prores2d,self.solveres2d,self.epires2d = self.prores2d_nojit,self.solveres2d_nojit,self.epires2d_nojit

                
        values = databank.values.copy()
        outvalues = values.copy() 

        res_col = [databank.columns.get_loc(c) for c in self.solveorder]
        
       
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
        endtimesetup=time.time()
    
        starttime=time.time()
        self.stackrows=[databank.index.get_loc(p) for p in sol_periode]
        with ttimer(f'\nres calculation',timeit) as xxtt:
                for row in self.stackrows:
                    self.prores2d(values, outvalues, row ,  alfa )
                    self.solveres2d(values, outvalues, row ,  alfa )
                    self.epires2d(values, outvalues, row ,  alfa )
            
        outdf =  pd.DataFrame(outvalues,index=databank.index,columns=databank.columns)  
        
        if stats:
            numberfloats = self.calculate_freq[-1][1]*len(self.stackrows)
            endtime = time.time()
            self.simtime = endtime-starttime
            self.setuptime = endtimesetup - starttimesetup
            print(f'Setup time (seconds)                 :{self.setuptime:>15,.2f}')
            print(f'Foating point operations             :{numberfloats:>15,}')
            print(f'Simulation time (seconds)            :{self.simtime:>15,.2f}')
                
            
        if not silent : print (self.name + ' solved  ')
        return outdf 

    def control(self,databank,targets,instruments,silent=True,ljit=0, 
                maxiter = 30,**kwargs):
        self.t_i = targets_instruments(databank,targets,instruments,self,silent=silent,
                 DefaultImpuls=0.01,defaultconv=0.001,nonlin=False, maxiter = maxiter)
        res = self.t_i()
        return res


class newton_diff():
    ''' Class to handle newron solving 
    
    ''' 
    def __init__(self, mmodel, df = None , endovar = None,onlyendocur=False, 
                 timeit=False, silent = True, forcenum=False,per='',ljit=0,nchunk=None):
        self.df          = df if type(df) == pd.DataFrame else mmodel.lastdf 
        self.endovar     = sorted(mmodel.endogene if endovar == None else endovar)
        self.placdic = {v : i for i,v in enumerate(self.endovar)}
        self.mmodel      = mmodel
        self.onlyendocur = onlyendocur
        self.silent = silent
        self.maxdif = 9999999999999
        self.forcenum = forcenum 
        self.timeit= timeit 
        self.per=per
        self.ljit=ljit
        self.nchunk = nchunk
        print(f' Initialize {self.ljit}')
        self.diffendocur = self.modeldiff()
        self.diff_model = self.get_diffmodel()
            
    def modeldiff(self):
        ''' Differentiate relations for self.enovar with respect to endogeneous variable 
        The result is placed in a dictory in the model instanse: model.diffendocur
        '''
    
        def findallvar(model,v):
            '''Finds all endogenous variables which is on the right side of = in the expresion for variable v
            lagged variables are included '''
            terms= self.mmodel.allvar[v]['terms'][model.allvar[v]['assigpos']:-1]
            if self.onlyendocur :
                rhsvar={(nt.var+('('+nt.lag+')' if nt.lag != '' else '')) for nt in terms if nt.var and nt.lag == '' and nt.var in self.endovar}
            else:
                rhsvar={(nt.var+('('+nt.lag+')' if nt.lag != '' else '')) for nt in terms if nt.var and nt.var in self.endovar}
            var2=sorted(list(rhsvar))
            return var2

        with ttimer('Find espressions for partial derivatives',self.timeit):
            diffendocur={} #defaultdict(defaultdict) #here we wanmt to store the derivativs
            i=0
            for nvar,v in enumerate(self.endovar):
                if nvar >= self.maxdif:
                    break 
                if not self.silent: 
                    print(f'Now differentiating {v} {nvar}')
                    
                endocur = findallvar(self.mmodel,v)
                    
                diffendocur[v]={}
                t=self.mmodel.allvar[v]['frml'].upper()
                a,fr,n,udtryk=split_frml(t)
                udtryk=udtryk
                udtryk=re.sub(r'LOG\(','log(',udtryk) # sympy uses lover case for log and exp 
                udtryk=re.sub(r'EXP\(','exp(',udtryk)
                lhs,rhs=udtryk.split('=',1)
                try:
                    if not self.forcenum:
                        kat=sympify(rhs[0:-1], md._clash) # we take the the $ out _clash1 makes I is not taken as imiganary 
                except:
                    print('* Problem sympify ',lhs,'=',rhs[0:-1])
                for rhv in endocur:
                    try:
                        if not self.forcenum:
                            ud=str(kat.diff(sympify(rhv,md._clash)))
                            ud = re.sub(pt.namepat+r'(?:(\()([0-9])(\)))',r'\g<1>\g<2>+\g<3>\g<4>',ud)

                        if self.forcenum or 'Derivative(' in ud :
                            ud = md.numdif(self.mmodel,v,rhv,silent=self.silent)
                            if not self.silent: print('numdif of {rhv}')
                        diffendocur[v.upper()][rhv.upper()]=ud
        
                    except:
                        print('we have a serous problem deriving:',lhs,'|',rhv,'\n',lhs,'=',rhs)
                    i+=1
        if not self.silent:        
            print('Model                           :',self.mmodel.name)
            print('Number of endogeneus variables  :',len(diffendocur))
            print('Number of derivatives           :',i) 
        return diffendocur

    def get_diffmodel(self):
        ''' Returns a model which calculates the partial derivatives of a model'''
        
        def makelag(var):
            vterm = udtryk_parse(var)[0]
            if vterm.lag:
                if vterm.lag[0] == '-':
                    return f'{vterm.var}__lag__{vterm.lag[1:]}'
                elif vterm.lag[0] == '+':
                    return f'{vterm.var}__lead__{vterm.lag[1:]}'
                else:
                    return f'{vterm.var}__per__{vterm.lag}'
            else:
                return f'{vterm.var}__lag__0'
            
        with ttimer('Generates a model which calculatews the derivatives for a model',self.timeit):
            out = '\n'.join([f'{lhsvar}__p__{makelag(rhsvar)} = {self.diffendocur[lhsvar][rhsvar]}  '
                    for lhsvar in sorted(self.diffendocur)
                      for rhsvar in sorted(self.diffendocur[lhsvar])
                    ] )
            dmodel = newmodel(out,funks=self.mmodel.funks,straight=True,
           modelname=self.mmodel.name +' Derivatives '+ ' no lags and leads' if self.onlyendocur else ' all lags and leads')
        return dmodel 
    

    def get_diff_mat_tot(self,periode='',df=None):
        '''makes a derivative matrices for all lags '''
        
        def get_lagnr(l):
            ''' extract lag/lead from variable name and returns a signed lag (leads are positive'''
            return int('-'*(l.split('__')[0]=='LAG') + l.split('__')[1])
        
        
        def get_elm(vartuples,i):
            ''' returns a list of lags  list of tupels '''
            return [v[i] for v in vartuples]
            
        _per = self.mmodel.current_per  
        _df = self.df if type(df)  != pd.DataFrame else df  
    
        self.diff_model.current_per = _per     
        with ttimer('calculate derivatives',self.timeit):
            self.difres = self.diff_model.res2d(_df,silent=self.silent,stats=0,ljit=self.ljit,chunk=self.nchunk).loc[_per,self.diff_model.endogene]
        with ttimer('Prepare wide input to sparse matrix',self.timeit):
        
            cname = namedtuple('cname','var,pvar,lag')
            self.coltup = [cname(i.rsplit('__P__',1)[0], 
                            i.rsplit('__P__',1)[1].split('__',1)[0],
                  get_lagnr(i.rsplit('__P__',1)[1].split('__',1)[1])) 
                   for i in self.difres.columns]
            
            self.coltupnum = [(self.placdic[var],self.placdic[pvar],lag) 
                               for var,pvar,lag in self.coltup]
            
            self.difres.columns = self.coltupnum
            numbers = [i for i,n in enumerate(self.difres.index)]
            maxnumber = max(numbers)
            nvar = len(self.endovar)
            self.difres.loc[:,'number'] = numbers
            
        with ttimer('melt the wide input to sparse matrix',self.timeit):
            self.dmelt = self.difres.melt(id_vars='number')
            
        with ttimer('assign tall input to sparse matrix',self.timeit):

            self.dmelt = self.dmelt.assign(var = lambda x: get_elm(x.variable,0),
                                          pvar = lambda x: get_elm(x.variable,1),
                                          lag  = lambda x: get_elm(x.variable,2)) 
        with ttimer('eval tall input to sparse matrix',self.timeit):
            self.dmelt = self.dmelt.eval('''\
            keep = (@maxnumber >= lag+number) & (lag+number  >=0)
            row = number * @nvar + var
            col = (number+lag) *@nvar +pvar ''')
            size = nvar*len(numbers)
        with ttimer('query  tall input to sparse matrix',self.timeit):
            self.dmelt = self.dmelt.query('keep')
        with ttimer('Prepare numpy input to sparse matrix',self.timeit):

            diagvalues = np.full(size,-1.0)
            diagindex = np.array(range(size))
    #csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
            values =  np.concatenate((self.dmelt.value.values,diagvalues))  
            indicies = (np.concatenate((self.dmelt.row.values,diagindex)),
                        np.concatenate((self.dmelt.col.values,diagindex)))
        with ttimer('Create sparce matrix for all periods',self.timeit): 
            self.stacked = sp.sparse.csc_matrix((values,indicies ), 
                    shape=(size, size))

        return self.stacked

    def get_diff_mat_1per(self,periode='',df=None):
        '''makes a derivative matrices for all lags '''
        
        def get_lagnr(l):
            ''' extract lag/lead from variable name and returns a signed lag (leads are positive'''
            return int('-'*(l.split('__')[0]=='LAG') + l.split('__')[1])
        
        
        def get_elm(vartuples,i):
            ''' returns a list of lags  list of tupels '''
            return [v[i] for v in vartuples]
            
        _per = self.mmodel.current_per if periode == '' else periode 
        _df = self.df if type(df)  != pd.DataFrame else df  
    
        self.diff_model.current_per = _per     
        with ttimer('calculate derivatives',self.timeit):
            self.difres = self.diff_model.res2d(_df,silent=self.silent,stats=0,ljit=self.ljit,chunk=self.nchunk).loc[_per,self.diff_model.endogene]
        with ttimer('Prepare wide input to sparse matrix',self.timeit):
        
            cname = namedtuple('cname','var,pvar,lag')
            self.coltup = [cname(i.rsplit('__P__',1)[0], 
                            i.rsplit('__P__',1)[1].split('__',1)[0],
                  get_lagnr(i.rsplit('__P__',1)[1].split('__',1)[1])) 
                   for i in self.difres.columns]
            
            self.coltupnum = [(self.placdic[var],self.placdic[pvar],lag) 
                               for var,pvar,lag in self.coltup]
            
            self.difres.columns = self.coltupnum
            numbers = [n for i,n in enumerate(self.difres.index)]
#            maxnumber = max(numbers)
            nvar = len(self.endovar)
            self.difres.loc[:,'number'] = numbers
            
        with ttimer('melt the wide input to sparse matrix',self.timeit):
            self.dmelt = self.difres.melt(id_vars='number')
            
        with ttimer('assign tall input to sparse matrix',self.timeit):

            self.dmelt = self.dmelt.assign(var = lambda x: get_elm(x.variable,0),
                                          pvar = lambda x: get_elm(x.variable,1),
                                          lag  = lambda x: get_elm(x.variable,2)) 
        with ttimer('eval tall input to sparse matrix',self.timeit):
            self.dmelt = self.dmelt.eval('''\
            keep = 1
            row =  var
            col = pvar ''')
        with ttimer('Prepare numpy input to sparse matrix',self.timeit):
            outdic = {}
            diagvalues = np.full(nvar,-1.0)
            diagindex = np.array(range(nvar))
            grouped = self.dmelt.groupby(by='number')  
            for per,df in grouped:
#                print(per)
#                print(df)
    #csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) 
                values =  np.concatenate((df.value.values,diagvalues))  
                indicies = (np.concatenate((df.row.values,diagindex)),
                            np.concatenate((df.col.values,diagindex)))
                outdic[per] = sp.sparse.csc_matrix((values,indicies ), 
                    shape=(nvar, nvar))

        return outdic

    def get_solve1perlu(self,df='',periode=''):
#        if update or not hasattr(self,'stacked'):
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.ludic = {p : sp.linalg.lu_factor(jac.toarray()) for p,jac in self.jacsparsedic.items()}
        self.solveludic = {p: lambda distance : sp.linalg.lu_solve(lu,distance) for p,lu in self.ludic.items()}
        return self.solveludic
    
    def get_solve1per(self,df='',periode=''):
#        if update or not hasattr(self,'stacked'):
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.solvelusparsedic = {p: sp.sparse.linalg.factorized(jac) for p,jac in self.jacsparsedic.items()}
        return self.solvelusparsedic

    def get_jacdf(self,df='',periode=''):
        self.jacsparsedic = self.get_diff_mat_1per(df=df,periode=periode)
        self.jacdfdic = {p: pd.DataFrame(jac.toarray(),columns=self.endovar,index=self.endovar) for p,jac in self.jacsparsedic.items()}
        return self.jacdfdic
    
    def get_solvestacked(self,df=''):
#        if update or not hasattr(self,'stacked'):
        self.stacked = self.get_diff_mat_tot(df=df)
        self.solvestacked = sp.sparse.linalg.factorized(self.stacked)
        return self.solvestacked
    
    def get_solvestacked_it(self,df='',solver = sp.sparse.linalg.bicg):
#        if update or not hasattr(self,'stacked'):
        self.stacked = self.get_diff_mat_tot(df=df)
        
        def solvestacked_it(b):
            return  solver(self.stacked,b)[0] 
        
        return solvestacked_it


class newvis(vis):
    
    pass 

def f(a):
    return 42

if __name__ == '__main__':
    
    
    #%%
    #this is for testing 
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        ftest = ''' 
        FRMl <>  ii = TY(-1)+c(-1)+Z*c(+1) $
        frml <>  c=0.8*yd+log(1) $
        frml <>  d = c +2*ii(-1) $
       frml <>  c2=0.8*yd+log(1) $
        frml <>  d2 = c + 42*ii $
       frml <>  c3=0.8*yd+log(1) $
        frml <>  d3 = c +ii $
 '''
        m2=newmodel(ftest,funks=[f],straight=True,modelname='m2 testmodel')
        df2=insertModelVar(df2,m2)
        xx = m2.xgenr(df2)
        m2.lastdf = xx
        m2.res2d(df2,ljit=0,chunk=2,debug=1)
        zz = m2.make_res_text2d_nojit
        #%%
        nn = newton_diff(m2,df=df2,timeit=0,onlyendocur=1)
        mat_dif = nn.get_diff_mat_tot(df=xx)
        md1 = mat_dif.toarray()
 #%%       
        mat_dif2 = nn.get_diff_mat_1per(df=xx)
        md2 = {p : sm.toarray() for p,sm in mat_dif2.items()}
        solvedic = nn.get_solve1per()
        xr = nn.diff_model.make_res_text2d_nojit
        #%%     
        m2._vis = newvis 
        cc1 = m2.outsolve2dcunk(df2,type='res')
        #%%
        if 0:
#            m2(df)
            dfr1=m2(df2,antal=10,fairantal=1,debug=1,conv='Y',ldumpvar=0,dumpvar=['C','Y'],stats=False,ljit=0,chunk=2)
            dd = m2.make_los_text1d
            assert 1==1
#           print(m2.make_los_text2d)
            #%%
            m2.use_preorder=0
            dfr1=m2(df2,antal=10,fairantal=1,debug=1,conv='Y',ldumpvar=1,dumpvar=['C','Y'],stats=True,ljit=1)
#%%            
            m2.Y.explain(select=True,showatt=True,HR=False,up=1)
    #        g  = m2.ximpact('Y',select=True,showatt=True,lag=True,pdf=0)
            m2.Y.explain(select=0,up=2)
    #        m2.Y.dekomp(lprint=1)
    #        m2.Y.draw(all=1)
    #        m2.vis('dog*').dif.heat()
            x= m2.Y.show
            m2['I*'].box()
            assert 1==1   
 #%% 
        if 1:           
            def test(model):
                for b,t in zip(model.strongblock,model.strongtype):
                    pre = {v for v,indegree in model.endograph.in_degree(b) 
                                if indegree == 0}
                    epi = {v for v,outdegree in model.endograph.out_degree(b) 
                                if outdegree == 0}
                    print(f'{t:20} {len(b):6}  In pre: {len(pre):4}  In epi: {len(epi):4}')
                    
