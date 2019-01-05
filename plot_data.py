import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#set font size of labels on matplotlib plots
plt.rc('font', size=16)

#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

output_file = 'clustered_data.csv'
cluster_data = pd.read_csv(output_file, delimiter=',')
cluster_data.columns = ['x', 'y', 'clusters']
facet = sns.lmplot(data=cluster_data, x='x', y='y', hue='clusters',
                   fit_reg=False, legend=False)
plt.show()
