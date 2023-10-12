#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.DataFrame({'y': [0.0000, 0.0000, 0.3333, 0.3667, 0.3333, 0.3712, 0.4130, 0.4315, 0.4414,
#         0.6157], 'x':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]})

# sns.barplot(data=data, x='x', y='y', color= 'orange')
# data_line = pd.DataFrame({
#     'x': np.arange(0, 1.1, 0.1),
#     'y': np.arange(0, 1.1, 0.1)
# })
sns.lineplot(x= np.arange(0, 1.1, 0.1), y=np.arange(0, 1.1, 0.1), color= 'blue', style=True, dashes=[(2,2)])
plt.show()
# %%
import seaborn as sns
import numpy as np
x = np.linspace(0, np.pi, 111)
y = np.sin(x)
sns.lineplot(x, y, style=True, dashes=[(2,2)])
# %%
def plot_reliability_diagrams(bins_accuracy, n_bins = 10):
    x=np.arange(0,1,1.0/n_bins)
    accuracy=bins_accuracy
    fig,ax=plt.subplots(figsize=(8,8))
    tick=np.arange(0,1.1,1.0/n_bins)
    # plt.bar(x,y,width=0.1,align='edge',color='red')
    plt.bar(x,accuracy,width=0.1,align='edge',color='blue',alpha=0.4,edgecolor='black',linewidth=0.5)
    line,=plt.plot([0,1],[0,1],ls='--',color='grey')
    line.set_dashes((3,7))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    plt.grid(True,color='black',linewidth=0.5,linestyle='--',dashes=(5,15))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.show()
plot_reliability_diagrams()
# fig.savefig('../output_fig/total_au_cali.jpg', bbox_inches='tight')
# %%
y
# %%
