import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2) 

df = pd.read_csv('results.csv')

print(df)

df.Score = df.Score * 10

sns_plot = sns.lineplot(data=df, y="Score", x="Time", hue='Model', color='blue', linewidth=4)
fig = sns_plot.get_figure()

sns_plot.set(xlabel="Time (Hours)", ylabel = "Score")
sns_plot.set_xticks(range(10)) 
sns_plot.set_xticklabels(['0','1','2','3','4','5','6','7', '8', '9'])

sns_plot.set_yticks(range(11)) 
sns_plot.set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7', '0.8', '0.9', '1'])

fig.savefig('ClassifierComp.png')
plt.show()