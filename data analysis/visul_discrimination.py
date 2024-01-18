import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)


data = pd.read_csv("item_parameter.csv")
label = data['format']
word = data['wordId']
dis = data['a']


data.plot.scatter(x='wordId', y='a', c='format',cmap='coolwarm', colorbar=True)
plt.show()