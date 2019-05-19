# -*- coding: utf-8 -*-
"""
Module for making attribution analysis of a model. 

The main function is attribution 

Created on Wed May 31 08:50:51 2017

@author: hanseni

"""


import pandas as pd
import fnmatch 

from modelclass import ttimer 
import modelclass as mc 


def attribution(model,experiments,start='2016q1',end='2018q4',save='',maxexp=1000,showtime=False,
                summaryvar=['RCET1__*','AGG_RCET1__*','PD__[!Q]*','LOGITPD__[!Q]*']
                ,silent=False):
    """ Calculates an attribution analysis on a model 
    accepts a dictionary with experiments. the key is experiment name, the value is a list 
    of variables which has to be reset to the values in the baseline dataframe. """  
    summaryout = model.vlist(summaryvar)
    adverseny = model.lastdf
    base = model.basedf
    adverse0=adverseny[summaryout].loc[start:,:].copy() 
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
                silent=True)[summaryout].loc[start:,:]
            adverseny[var] = temp
#            adverseny = mc.upddf(adverseny,temp)

        difret = {e : adverse0-ret[e]           for e in ret}
    
    df = pd.concat([difret[v] for v in difret],keys=difret.keys()).T
    if save:
         df.to_pickle('data\\' +save +r'.pc')    
         
    model.save = modelsave # restore the state of model.save 
    return df

def ilist(df,pat):
    '''returns a list of variable in the index of a dataframe, the pattern can be a list of patterns'''
    ipat = pat if isinstance(pat,list) else [pat]
    return [v for  p in ipat for v in sorted(fnmatch.filter(df.index,p.upper()))] 
    
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

def AggImpact(impact):
    """ Calculates the sum of impacts and place in the last column""" 
    asum= impact.sum(axis=1)
    asum.name = '_Sum'
    aout = pd.concat([impact,asum],axis=1)
    return aout 


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
        
