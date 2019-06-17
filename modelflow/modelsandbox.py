# -*- coding: utf-8 -*-
"""

This is a module for testing new features of the model class, but in a smaler file. 

Created on Sat Sep 29 06:03:35 2018

@author: hanseni
"""

import pandas as pd
from collections import defaultdict, namedtuple
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

import sys  
import time





from modelclass import model, ttimer, insertModelVar
from modelvis import vis

import modelpattern as pt
import modelmanipulation as mp
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
            if self.save:
                if self.previousbase and hasattr(self,'lastdf'):
                    self.basedf = self.lastdf.copy(deep=True)
                
            if kwargs.get('sim2',False):
                outdf = self.sim2d(*args, **kwargs )   
            else: 
                outdf = self.sim1d( *args, **kwargs) 
    
            if self.save:
                if (not hasattr(self,'basedf')) or kwargs.get('setbase',False) : self.basedf = outdf.copy(deep=True) 
                if kwargs.get('setlast',True)                                  : self.lastdf = outdf.copy(deep=True)
        
            return outdf

  
    @property 
    def showstartnr(self):
        self.findpos()
        variabler=[x for x in sorted(self.allvar.keys())]
        return {v:self.allvar[v]['startnr'] for v in variabler}

        
        
       
    def sim2d(self, databank, start='', slut='', silent=0,samedata=0,alfa=1.0,stats=False,first_test=1,
              antal=1,conv=[],absconv=0.01,relconv=0.00001,
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=1_000_000,ljit=False, 
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
        if not samedata or not hasattr(self,'solve2d') :
           if (not hasattr(self,'solve2d')) or (not self.eqcolumns(self.genrcolumns,databank.columns)):
                databank=insertModelVar(databank,self)   # fill all Missing value with 0.0 
                for i in [j for j in self.allvar.keys() if self.allvar[j]['matrix']]:
                    databank.loc[:,i]=databank.loc[:,i].astype('O')   #  Make sure columns with matrixes are of this type 

                self.make_los_text2d =  self.outsolve2dcunk(databank,chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                exec(self.make_los_text2d,globals())  # creates the los function
                self.pro2d,self.solve2d,self.epi2d  = make_los(self.funks,self.errfunk)
                
        values = databank.values.copy()  # 
        self.genrcolumns = databank.columns.copy()  
        self.genrindex   = databank.index.copy()  
       
        convvar = [conv.upper()] if isinstance(conv,str) else [c.upper() for c in conv] if conv != [] else list(self.endogene)  
        convplace=[databank.columns.get_loc(c) for c in convvar] # this is how convergence is measured  
        convergence = True
 
        if ldumpvar:
            self.dumplist = []
            self.dump = convvar if dumpvar == [] else self.vlist(dumpvar)
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
                self.pro2d(values, values,  row ,  alfa )
                for iteration in range(antal):
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
                self.epi2d(values, values, row ,  alfa )

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
    


    def outsolve2dcunk(self,databank, debug=1,chunk=30,ljit=False):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        columnsnr=self.get_columnsnr(databank)
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
       
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
            

        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+'@jit("(f8[:,:],f8[:,:],i8,f8)",fastmath=True)\n')
            fib1.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
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
                                              ljit=ljit,oldeqs=neweqs)
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
                
            fib2.append(short + 'def '+name+'(values,outvalues,row,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(values,outvalues,row,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  make_gaussline2  
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
        fib1.append(short + 'from modeluserfunk import '+(', '.join(pt.userfunk)).lower()+'\n')
        fib1.append(short + 'from modelBLfunk import '+(', '.join(pt.BLfunk)).lower()+'\n')
        funktext =  [short+f.__name__ + ' = funks['+str(i)+']\n' for i,f in enumerate(self.funks)]      
        fib1.extend(funktext)
        
        
        
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
              dumpvar=[],ldumpvar=False,dumpwith=15,dumpdecimal=5,chunk=1_000_000,ljit=False, 
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
                       self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                       exec(self.make_los_text1d,globals())  # creates the los function
                       self.pro1d_jit,self.solve1d_jit,self.epi1d_jit  = make_los(self.funks,self.errfunk)
                   this_pro1d,this_solve1d,this_epi1d = self.pro1d_jit,self.solve1d_jit,self.epi1d_jit
            else:  
                    if not hasattr(self,'solve1d'): 
                        self.make_los_text1d =  self.outsolve1dcunk(chunk=chunk,ljit=ljit, debug=kwargs.get('debug',1))
                        exec(self.make_los_text1d,globals())  # creates the los function
                        self.pro1d,self.solve1d,self.epi1d  = make_los(self.funks,self.errfunk1d)
 
                    this_pro1d,this_solve1d,this_epi1d = self.pro1d,self.solve1d,self.epi1d                

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
                print(f'Fair-Taylor iteration: {fairiteration}')
            for self.periode in sol_periode:
                row=databank.index.get_loc(self.periode)
                self.row_ = row
                a=self.stuff3(values,row,ljit)
#                  
                if ldumpvar:
                    self.dumplist.append([fairiteration,self.periode,int(0)]+[a[p] 
                        for p in dumpplac])
    
                itbefore = [a[c] for c in convplace] 
                this_pro1d(a,  alfa )
                for iteration in range(antal):
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
                this_epi1d(a ,  alfa )
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
    
    
    def outsolve1dcunk(self,debug=0,chunk=None,ljit=False):
        ''' takes a list of terms and translates to a evaluater function called los
        
        The model axcess the data through:Dataframe.value[rowindex+lag,coloumnindex] which is very efficient 

        '''        
        short,long,longer = 4*' ',8*' ',12 *' '
        self.findpos()
        if ljit:
            thisdebug = False
        else: 
            thisdebug = debug 
        
        
           

        def makeafunk(name,order,linemake,chunknumber,debug=False,overhead = 0 ,oldeqs=0,nodamp=False,ljit=False):
            ''' creates the source of  an evaluation function 
            keeps tap of how many equations and lines is in the functions abowe. 
            This allows the errorfunction to retriewe the variable for which a math error is thrown 
            
            ''' 
            fib1=[]
            fib2=[]
            
            if ljit:
                fib1.append((short+'print("'+f"Compiling chunk {chunknumber}     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib1.append(short+'@jit("(f8[:],f8)",fastmath=True,cache=False)\n')
            fib1.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib1.append(long + 'outvalues = values \n')
            if debug:
                fib1.append(long+'try :\n')
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
                                              ljit=ljit,oldeqs=neweqs)
                fib.extend(lines)
                newoverhead = head 
                neweqs = eques
            if ljit:
                fib2.append((short+'print("'+f"Compiling a mastersolver     "+'",time.strftime("%H:%M:%S")) \n') if ljit else '')
                fib2.append(short+'@jit("(f8[:],f8)",fastmath=True,cache=False)\n')
                 
            fib2.append(short + 'def '+name+'(a,alfa=1.0):\n')
#            fib2.append(long + 'outvalues = values \n')
            tt =[long+name+str(i)+'(a,alfa=alfa)\n'   for (i,ch) in enumerate(orderlist)]
            fib2.extend(tt )
            fib2.append(long+'return  \n')

            return fib+fib2,newoverhead+len(fib2),neweqs
                
                
            
        linemake =  self.make_gaussline  
        fib2 =[]
        fib1 =     ['def make_los(funks=[],errorfunk=None):\n']
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
    
    def make_resline2d(self,vx):
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
                    out.append('bvalues[row'+t.lag+','+str(columnsnr[t.var])+']' )              
                else:
                    out.append('values[row'+t.lag+','+str(columnsnr[t.var])+']' )              
        res = ''.join(out)
        return res
  
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

class newvis(vis):
    
    pass 

def f(a):
    return 42

if __name__ == '__main__':
    #%%
    #this is for testing 
        df = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df1 = pd.DataFrame({'Z': [1., 2., 3,4] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=   [2017,2018,2019,2020])
        df2 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df3 = pd.DataFrame({'Z':[1., 22., 33,43] , 'TY':[10.,20.,30.,40.] ,'YD':[10.,20.,30.,40.]},index=[2017,2018,2019,2020])
        df4 = pd.DataFrame({'Z':[ 223., 333] , 'TY':[203.,303.] },index=[2018,2019])
        ftest = ''' 
        FRMl <>  ii = x+z(-1)+log(1) $
        frml <>  c=0.8*yd+log(1) $
        FRMl <z>  i = ii+iy +log(1) $ 
        FRMl <>  x = f(2) $ 
        FRMl <>  y = c(-1) + i + x+ i(-1)+log(-1)$ 
        FRMl <>  yX =  1.0*y(+1) $
        FRML <>  dogplace = y *4 $'''
        
        m2=newmodel(ftest,funks=[f],straight=True)
        m2._vis = newvis 
        if 1:
#            m2(df)
            df2=insertModelVar(df2,m2)
            cc = m2.outsolve2dcunk(df2)
            dfr1=m2.sim1d(df2,antal=10,fairantal=1,debug=1,conv='Y',ldumpvar=0,dumpvar=['C','Y'],stats=False,ljit=0,chunk=2)
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
                    
