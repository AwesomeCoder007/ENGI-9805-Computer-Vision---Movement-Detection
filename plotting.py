"""
Course : ENGI 9805 Computer Vision
Project : Motion detector
Team members : Utkarsh Trivedi , Sona Pujari , Shujhana Mostafa
Date Last Modified : 02/04/2020
Version : 1.9

"""

# Importing from motion_detector and bokeh libraries
from motion_detector import df
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

# formatting DataFrame according to our usage
df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Assigning DataFrame to ColumnDataSource
cds=ColumnDataSource(df)

# using figure function of bokeh libraries
p=figure(x_axis_type='datetime',height=100,width=800,sizing_mode='scale_width',title="Motion Graph")
p.yaxis.minor_tick_line_color=None

# desired num tick used to work in jupyter book below version 0.7 but gives error in latest
#So just plotted the graph without using it works okay without it
#p.ygrid[0].ticker.desired_num_ticks=1

# using hover tool for better optimization of graph
hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
p.add_tools(hover)

# Assigning values on top,bottom,left,right for ColumnDataSource
q=p.quad(left="Start",right="End",bottom=0,top=1,color="green",source=cds)

output_file("Graph1.html")
show(p)
