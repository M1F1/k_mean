import matplotlib.pyplot as plt
import numpy as np

output_file = 'clustered_data.csv'
cluster_data = pd.read_csv(output_file, delimiter=',')
cluster_data.columns = ['x', 'y', 'clusters']
facet = sns.lmplot(data=cluster_data, x='x', y='y', hue='clusters',
                   fit_reg=False, legend=False)
plt.show()
