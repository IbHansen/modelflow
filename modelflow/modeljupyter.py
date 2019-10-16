# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:26:13 2019

@author: hanseni
"""

import pandas as pd
import  ipywidgets as widgets  
from IPython.display import display, clear_output
import matplotlib.pylab  as plt 
import seaborn as sns
#import qgrid 

import sys
sys. path.append('modelflow/')
 

def vis_alt3(dfs,model,title='Show variables',basename='Baseline',altname='Alternative',trans={}):
    ''' display tabbed widget with results from different dataframes, usuallly 2 but more can be shown
    
    dfs is a list of dataframes. They should be of same dimensionalities 
    
    '''
    avar = dfs[0].columns
    outlist =     [widgets.Output() for var in avar]
    outdiflist =  [widgets.Output() for var in avar]    
    deplist =     [widgets.Output() for var in avar]
    reslist =     [widgets.Output() for var in avar]
    attlist =     [widgets.Output() for var in avar]
    varouttablist =  [widgets.Tab(children = [out,outdif,att,dep,res]) 
                      for out,outdif,att,dep,res  in zip(outlist,outdiflist,attlist,deplist,reslist)]
    for var,varouttab in zip(avar,varouttablist):
        for i,tabtext in enumerate(['Level','Change','Attribution','Dependencies','Results']):
            varouttab.set_title(i,tabtext)

    controllist = [widgets.VBox([varouttab]) for varouttab in varouttablist]
    tab = widgets.Tab(children = controllist)
    for i,(var,out) in enumerate(zip(avar,controllist)):
        tab.set_title(i, var) 
    
    def showvar(b):
        sel = b['new']
        out = outlist[sel]
        outdif = outdiflist[sel]
        dep = deplist[sel]
        res = reslist[sel]
        att = attlist[sel]
        var = avar[sel]
        with out:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
#            ax.spines['top'].set_visible(False)
            for i,df in enumerate(dfs):
                data = df.loc[:,var]
                data.plot(ax=ax,legend=False,fontsize=14)
                x_pos = data.index[-1]
                if i == 0:
                    basevalue = data.values[-1]
                else:
                    alt=i if len(dfs) >= 3 else ''
                    if (abs(data.values[-1]-basevalue) < 0.01 or 
                      (abs(basevalue) > 100. and (1.-abs(data.values[-1]/basevalue) < 0.008))):
                        if i == 1:
                            ax.text(x_pos, data.values[-1]  ,f' {basename} and {altname}{alt}',fontsize=14)
                    else:
                        ax.text(x_pos, data.values[-1]  ,f' {altname}{alt}',fontsize=14)
                        ax.text(x_pos, basevalue  ,f' {basename}',fontsize=14)
            plt.show(fig)
        with outdif:
            clear_output()
            fig,ax = plt.subplots(figsize=(10,6))
            ax.set_title(trans.get(var,var),fontsize=14)
            ax.spines['right'].set_visible(False)
            for i,df in enumerate(dfs):
                if i == 0:
                    basedata = df.loc[:,var]
                    x_pos = data.index[-1]
                else:
                    data = df.loc[:,var]-basedata
                    data.plot(ax=ax,legend=False,fontsize=14)
                    x_pos = data.index[-1]
                    alt=i if len(dfs) >= 3 else ''
                    ax.text(x_pos, data.values[-1]  ,f' Impact of {altname}{alt}',fontsize=14)
            plt.show(fig)
        with dep:
            clear_output()
            model.draw(var,up=2,down=2,svg=1)
        with res:
            clear_output()
            out = model.get_values(var).T.rename(columns={'Base':basename,'Last':altname})
            out.style.set_caption(trans.get(var,var))
            print(trans.get(var,var))
            print(out)
        with att:
            clear_output()
            #model.smpl(N-20,N)
            if var in model.endogene:
                print(f'What explains the difference between the {basename} and the {altname} run ')
                print(model.allvar[var]['frml'])
                model.explain(var,up=0,dec=1,size=(9,12),svg=1,HR=0)
            else:
                print(f'{var} is exogeneous and attribution can not be performed')
            
    display(tab) 

    showvar({'new':0})
    tab.observe(showvar,'selected_index')
    return tab

# Define data extraction
def get_alt(mmodel,pat):
    ''' Retrieves variables matching pat from a model '''
    varnames = mmodel.vlist(pat)
    modelvar = mmodel.exogene | mmodel.endogene
    modelvarnames = list(dict.fromkeys([v for v in varnames if v in modelvar])) # to awoid dublicate names
    per = mmodel.current_per
    return [mmodel.basedf.loc[per,modelvarnames],mmodel.lastdf.loc[per,modelvarnames]]





def inputwidget(model,df,slidedef={},radiodef=[],checkdef=[],modelopt={},varpat='RFF XGDPN RFFMIN GFSRPN DMPTRSH XXIBDUMMY'
                 ,showout=1,trans={}):
    '''Creates an input widgets for updating variables 
    
    :df: Baseline dataframe 
    :slidedef: dict with definition of variables to be updated by slider
    :radiodef: dict of lists. each at first level defines a collection of radiobuttoms
               second level defines the text for each leved and the variable to set or reset to 0
    :varpat: the variables to show in the output widget
    :showout: 1 if the output widget is to be called '''
    
    lradiodef= len(radiodef)
    lslidedef = len(slidedef)
    lcheckdef = len(checkdef)
    basename ='Baseline'
    
    if lradiodef: 
        wradiolist = [widgets.RadioButtons(options=[i for i,j in cont],description=des,layout={'width':'70%'},
                                           style={'description_width':'37%'}) for des,cont in radiodef.items()]
        if len(wradiolist) <=2:
            wradio = widgets.HBox(wradiolist)
        else: 
            wradio = widgets.VBox(wradiolist)

            

# define slidesets 
    if lslidedef:     
        wexp  = widgets.Label(value="Input new parameter ",layout={'width':'41%'})
        walt  = widgets.Label(value="Alternative",layout={'width':'8%'})
        wbas  = widgets.Label(value="Baseline",layout={'width':'8%'})
        whead = widgets.HBox([wexp,walt,wbas])
        
        wset  = [widgets.FloatSlider(description=des,
                                    min=cont['min'],max=cont['max'],value=cont['value'],step=cont.get('step',0.01),
                                    layout={'width':'50%'},style={'description_width':'30%'})
                 for des,cont in slidedef.items()]
        waltval= [widgets.Label(value=f"{cont['value']:>.2f}",layout={'width':'8%'})
                  for des,cont  in slidedef.items()]
        wslide = [widgets.HBox([s,v]) for s,v in zip(wset,waltval)]
       
# cheklist  
    if lcheckdef:         
        wchecklist = [widgets.Checkbox(description=des,value=val)   for des,var,val in checkdef]
        wcheck  = widgets.HBox(wchecklist)   

# some buttons and text     
    wname = widgets.Text(value='Alternative',placeholder='Type something',description='Name of experiment:',
                        layout={'width':'30%'},style={'description_width':'50%'})
    wpat = widgets.Text(value=varpat,placeholder='Type something',description='Output variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
    winputstring = widgets.HBox([wname,wpat])
    
    wgo   = widgets.Button(description="Run")
    wreset   = widgets.Button(description="Reset to default")
    wsetbas   = widgets.Button(description="Use as baseline")
    wbut  = widgets.HBox([wgo,wreset,wsetbas])
    
    wvar = [whead]+wslide if lslidedef else []
    if lradiodef: wvar = wvar + [wradio]
    if lcheckdef: wvar = wvar + [wcheck]
        
    w     = widgets.VBox(wvar+[winputstring] +[wbut])

    # This function is run when the button is clecked 
    def run(b):
        mulstart       = model.basedf.copy()
        
        # First update from the sliders 
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                op = cont.get('op','=')
                var = cont['var']
                for var in cont['var'].split():
                    if  op == '+':
                        mulstart.loc[model.current_per,var]    =  mulstart.loc[model.current_per,var] + wset[i].value
                    elif op == '+impulse':    
                        mulstart.loc[model.current_per[0],var] =  mulstart.loc[model.current_per[0],var] + wset[i].value
                    elif op == '=':    
                        mulstart.loc[model.current_per,var] =   wset[i].value
                    elif op == '=impuse':    
                        mulstart.loc[model.current_per,var] =   wset[i].value
                    else:
                        print(f'Wrong operator in {cont}.\nNot updated')
                        
                
        # now  update from the radio buttons 
        if lradiodef:
            for wradio,(des,cont) in zip(wradiolist,radiodef.items()):
                print(des,wradio.value,wradio.index,cont[wradio.index])
                for v in cont:
                    mulstart.loc[model.current_per,v[1]] = 0.0
                mulstart.loc[model.current_per,cont[wradio.index][1]] = 1.0  
 
        if lcheckdef:           
            for box,(des,var,_) in zip(wchecklist,checkdef):
                mulstart.loc[model.current_per,var] = 1.0 * box.value

        #with out:
        clear_output()
        mul = model(mulstart,**modelopt)

        clear_output()
        display(w)
        #_ = mfrbus['XGDPN RFF RFFMIN GFSRPN'].dif.rename(trans).plot(colrow=1,sharey=0)
        if showout:
            a = vis_alt3(get_alt(model,wpat.value),model,basename=basename,altname=wname.value,trans=trans)

    def reset(b):
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                wset[i].value  =   cont['value']
            
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
            
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue

    def setbas(b):
        nonlocal basename
        model.basedf = model.lastdf.copy(deep=True)
        basename = wname.value
        
    # Assign the function to the button  
    wgo.on_click(run)
    wreset.on_click(reset)
    wsetbas.on_click(setbas)
    out = widgets.Output()
    
    return w
