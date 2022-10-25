import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

age_data = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat_data = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

plt.scatter(age_data, fat_data)
plt.title('scatter graph')
plt.xlabel('age')
plt.ylabel('fat')
plt.show()

stats.probplot(age_data, dist='norm', plot=plt)
plt.title('age-norm')
plt.show()

stats.probplot(fat_data, dist='norm', plot=plt)
plt.title('fat-norm')
plt.show()

d1 = sorted(age_data)
d2 = sorted(fat_data)
sns.regplot(x=pd.Series(d1), y=pd.Series(d2), ci=None, color='b', line_kws={'color': 'r'})
plt.title('age-fat')
plt.show()
