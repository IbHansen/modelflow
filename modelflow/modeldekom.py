# -*- coding: utf-8 -*-
"""
Module for making attribution analysis of a model. 

The main function is attribution 

Created on Wed May 31 08:50:51 2017

@author: hanseni

"""


import pandas as pd
import fnmatch 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy


from modelclass import ttimer 
import modelclass as mc 
import modeldekom as mk 
import modelvis as mv

idx= pd.IndexSlice
    
def attribution(model,experiments,start='',end='',save='',maxexp=10000,showtime=False,
                summaryvar=['*']
                ,silent=False,msilent=True):
    """ Calculates an attribution analysis on a model 
    accepts a dictionary with experiments. the key is experiment name, the value is a list 
    of variables which has to be reset to the values in the baseline dataframe. """  
    summaryout = model.vlist(summaryvar)
    adverseny = model.lastdf
    base = model.basedf
    adverse0=adverseny[summaryout].loc[start:end,:].copy() 
    ret={}
    modelsave = model.save  # save the state of model.save 
    model.save = False      # no need to save the experiments in each run 
    with ttimer('Total dekomp',showtime):
        for i,(e,var) in enumerate(experiments.items()):
            if i >= maxexp : break     # when we are testing 
            temp=adverseny[var].copy()
            if not silent:
                print(i,'Experiment :',e,'\n','Touching: \n', var)
            adverseny[var] = base[var]
            ret[e] = model(adverseny   ,start,end,samedata=True,
                silent=msilent)[summaryout].loc[start:end,:]
            adverseny[var] = temp
#            adverseny = mc.upddf(adverseny,temp)

        difret = {e : adverse0-ret[e]           for e in ret}
    
    df = pd.concat([difret[v] for v in difret],keys=difret.keys()).T
    if save:
         df.to_pickle('data\\' +save +r'.pc')    
         
    model.save = modelsave # restore the state of model.save 
    return df


def ilist(df,pat):
    '''returns a list of variable in the model matching the pattern, 
    the pattern can be a list of patterns of a sting with patterns seperated by 
    blanks
    
    This function operates on the index names of a dataframe. Relevant for attribution analysis
    '''
    if isinstance(pat,list):
           upat=pat
    else:
           upat = [pat]
           
    ipat = upat
    out = [v for  p in ipat for up in p.split() for v in sorted(fnmatch.filter(df.index,up.upper()))]  
    return out

    
def GetAllImpact(impact,sumaryvar):
    ''' get all the impact from at impact dataframe''' 
    exo = list({v for v,t in impact.columns})
    df = pd.concat([impact.loc[sumaryvar,c]  for c in exo],axis=1)
    df.columns = exo
    return df 

def GetSumImpact(impact,pat='PD__*'):
    """Gets the accumulated differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:].groupby(level=[0],axis=1).sum()   
    return a 

def GetLastImpact(impact,pat='RCET1__*'):
    """Gets the last differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:].groupby(level=[0],axis=1).last()   
    return a
 
def GetAllImpact(impact,pat='RCET1__*'):
    """Gets the last differences attributet to each impact group """ 
    a = impact.loc[ilist(impact,pat),:] 
    return a 

def GetOneImpact(impact,pat='RCET1__*',per=''):
    """Gets differences attributet to each impact group in period:per """ 
    a = impact.loc[ilist(impact,pat),idx[:,per]] 
    a.columns = [v[0] for v in a.columns]
    return a 

def AggImpact(impact):
    """ Calculates the sum of impacts and place in the last column
    
    This function is applied to the result iof a Get* function""" 
    asum= impact.sum(axis=1)
    asum.name = '_Sum'
    aout = pd.concat([impact,asum],axis=1)
    return aout 


class totdekomp():
    ''' Class to make modelvide attribution analysis 
    
    '''
    
    def __init__(self, model,summaryvar='*',desdic={} ):
       
       self.diffdf  = model.exodif()
       self.diffvar = self.diffdf.columns
       if len(self.diffvar) == 0:
           print('No variables to attribute to ')
           self.go = False 
       else: 
           self.go = True 
           self.experiments = {v:v for v in self.diffvar}
           self.model = model 
           self.start = self.model.current_per.tolist()[0]
           self.end = self.model.current_per.tolist()[-1]
           
           self.desdic = desdic       
           
           self.res = attribution(self.model,self.experiments,self.start,self.end,summaryvar=summaryvar,showtime=1,silent=1)
       
    def explain_last(self,pat='',top=0.9,title='Attribution last period'):  
        if self.go:         
            self.impact = mk.GetLastImpact(self.res,pat=pat).T
            fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title= title,desdic=self.desdic)
            return fig
   
    def explain_sum(self,pat='',top=0.9,title='Attribution, sum over all periods'): 
        if self.go:          
           self.impact = mk.GetSumImpact(self.res,pat=pat).T
           fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title=title,desdic=self.desdic )
           return fig
   
    def explain_per(self,pat='',per='',top=0.9,title='Attribution, for one periode'):   
        if self.go:        
           ntitle = f'{title}: {per}'
           self.impact = mk.GetOneImpact(self.res,pat=pat,per=per).T
           fig = mv.waterplot(self.impact,autosum=1,allsort=1,top=top,title=ntitle,desdic=self.desdic )
           return fig
   
            
    def explain_all(self,pat='',stacked=True,kind='bar',top=0.9,title='Attribution'): 
        if self.go:
            selected =   GetAllImpact(self.res,pat) 
            grouped = selected.stack().groupby(level=[0])
            fig, axis = plt.subplots(nrows=len(grouped),ncols=1,figsize=(10,5*len(grouped)),constrained_layout=False)
            width = 0.5  # the width of the barsser
            laxis = axis if isinstance(axis,numpy.ndarray) else [axis]
            for i,((name,dfatt),ax) in enumerate(zip(grouped,laxis)):
                dfatt.index = [i[1] for i in dfatt.index]
                dfatt.plot(ax=ax,kind=kind,stacked=stacked,title=self.desdic.get(name,name))
                ax.set_ylabel(name,fontsize='x-large')
                ax.set_xticklabels(dfatt.index.tolist(), rotation = 45,fontsize='x-large')
            fig.suptitle(title,fontsize=20)
            if 1:
                plt.tight_layout()
                fig.subplots_adjust(top=top)
        
            plt.show()

if __name__ == '__main__' :
    # running withe the mtotal model 
    if  ( not 'mtotal' in locals() ) or True:
        # get the model  
        with open(r"models\mtotal.fru", "r") as text_file:
            ftotal = text_file.read()
       
        #get the data 
        base0   = pd.read_pickle(r'data\base0.pc')    
        base    = pd.read_pickle(r'data\base.pc')    
        adve0   = pd.read_pickle(r'data\adve0.pc')     
    #%%    
        mtotal  = mc.model(ftotal)
        
#        prune(mtotal,base)
        #%%    
        baseny        = mtotal(base0   ,'2016q1','2018q4',samedata=False)
        adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
    #%%
    diff  = mtotal.exodif()   # exogeneous variables which are different between baseny and adverseny 
    #%%
    assert 1==2  # just for stopping in test situations 
    #%%  
    adverseny            = mtotal(adve0   ,'2016q1','2018q4',samedata=True)   # to makew sure we have the right adverse.
    countries            = {c.split('__')[2]  for c in diff.columns}  # list of countries 
    countryexperiments   = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in countries } # dic of experiments 
    assert len(diff.columns) == sum([len(c) for c in countryexperiments.values()]) , 'Not all exogeneous chocks variables are accountet for'
    countryimpact        = attribution(mtotal,countryexperiments,save='countryimpactxx',maxexp=30000,showtime = 1)    
    #%% 
    adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
    vartypes             = {c.split('__') [1]  for c in diff.columns}
    vartypeexperiments   = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in vartypes }
    assert len(diff.columns) == sum([len(c) for c in vartypeexperiments.values()]) , 'Not all exogeneous chocks variables are accountet for'
    vartypeimpact        = attribution(mtotal,vartypeexperiments,save='vartypeimpactxx',maxexp=3000,showtime=1)
    ##%%
    #adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
    #allexo                  = {c[7:14] for c in diff.columns} 
    #allexoexperiments       = {e: [c for c in diff.columns if ('__'+e+'__') in c]   for e in allexo }
    #allexoimpact            = attribution(mtotal,allexoexperiments,base,adverseny,save='allexoimpact',maxexp=2000)
#%% test of upddf    
    if 0:
        baseny        = mtotal(base0   ,'2016q1','2018q4',samedata=False)
        adverseny     = mtotal(adve0   ,'2016q1','2018q4',samedata=True)
#%%
        e = 'EE'     
        var = countryexperiments['EE']
        vardiff = diff[var]
        temp = adverseny[var].copy()
        adverseny[var] = baseny[var]
        temp2 = adverseny[var].copy()
        _ = mc.upddf(adverseny,temp) 
        adverseny[var] = temp
        temp3 = adverseny[var].copy()
        
