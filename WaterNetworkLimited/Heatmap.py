import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


# Defining index for the dataframe
idx = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

# Defining columns for the dataframe
cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

data_0 = pd.read_csv('df_action_0.csv', header=None)
v_min0=data_0.min().min()
v_max0=data_0.max().max()



data_1 = pd.read_csv('df_action_1.csv', header=None)
v_min1=data_1.min().min()
v_max1=data_1.max().max()

## What action will the aggent pick?
classification=np.where(data_1 < data_0,1,0)



## Plots
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
sn.heatmap(data_0,annot=False,vmin=v_min0,vmax=v_max0)
plt.title("Q values for action 0")

plt.subplot(1,2,2)
#plt.figure(figsize=(10,8))
sn.heatmap(data_1,annot=False,vmin=v_min1,vmax=v_max1)
plt.title("Q values for action 1")

plt.figure(figsize=(10,8))
sn.heatmap(classification,annot=False)
plt.title("The action that the agent will take")

plt.show()
