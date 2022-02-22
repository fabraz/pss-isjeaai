from metaflow import Flow,get_metadata,Run
print("Metadata",get_metadata())
from typing import List
import chart_studio.plotly as py
import plotly.graph_objects  as go
import plotly.express as ps
from reporting_data import *
from plotly.subplots import make_subplots
import math
import os 
import datetime
import itertools

def get_key_map(arr):
    finalmap = []
    for i in itertools.product(*arr):
        finalmap.append(i)
    return finalmap


def make_loss_plots(final_data_arr:List[ModelAnalytics],training_name='train',plot_name='loss')->go.Figure:
    """
    Makes the Loss Plot of all Agents within one single Plot. 
    :param training_name --> train | validation
    :param plot_name --> loss | batch_time | accuracy
    """
    import math
    final_arr = final_data_arr
    rows = math.ceil(len(final_arr)/3)+1
    last_time = datetime.datetime.now()
    index_queue = get_key_map([[i+1 for i in range(rows)],[1,2,3]]) # Make rows and columns. 
    last_row,last_col = index_queue[-1]
    loss_plot = make_subplots(
        rows=rows,\
        cols=3,\
        subplot_titles=[data.architecture for data in final_arr]+["All "+str(training_name).title()+"ing Losses In One Plot"], \
        specs=[ [{}, {},{}] for _ in range(rows-1) ]+ [[ {"colspan": 3}, None,None]]
    )
    for data in final_arr:
        row,col = index_queue.pop(0)
        train_loss_op = list(map(lambda x:sum(x[plot_name])/len(x[plot_name]),data.epoch_histories[training_name]))
        if not isinstance(train_loss_op,list):
            continue
        agent_name = data.architecture
        epochs = [j+1 for j in range(len(train_loss_op))]
        loss_plot.add_trace(go.Scatter(
                    x=epochs,
                    y=train_loss_op,
                    name=agent_name+"_"+training_name,
                    line=dict(width=1),
                    opacity=0.8),row=row,col=col)
     
        loss_plot.add_trace(go.Scatter(
                    x=epochs,
                    y=train_loss_op,
                    name=agent_name,
                    line=dict(width=1),
                    opacity=0.8),row=last_row,col=1)

        loss_plot.update_yaxes(title_text=plot_name.title(),row=row,col=col)
        loss_plot.update_xaxes(title_text="Epochs",row=row,col=col)
    
    loss_plot.update_yaxes(title_text=plot_name.title(),row=last_row,col=1)
    loss_plot.update_xaxes(title_text="Epochs",row=last_row,col=1)
    loss_plot.update_layout(title_text="Plot of "+str(training_name).title()+" Running "+plot_name.title()+" of All Models",height=2000,showlegend=False,width=2000)
    return loss_plot


def CountFrequency(my_list):       
    # Creating an empty dictionary  
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
    x ,y=[],[]
    for key, value in freq.items(): 
        x.append(key)
        y.append(value)
    return x,y
