import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Reading training dataset into a dataframe

train_data = load_iris()

#Adding column names to the dataframe

train_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','class_label']

#Reading test dataset into a dataframe

test_data = pd.read_csv('iris_test.data')
frames = [train_data,test_data]
data = pd.concat(frames)

classA = data[(data.class_label=='Iris-setosa')]
classB = data[(data.class_label=='Iris-versicolor')]
classB = data[(data.class_label=='Iris-versicolor')]


fig = plt.figure()
fig.suptitle('Sepal Length vs Sepal Width',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.scatter(classA['sepal_length'], classA['sepal_width'], c='g', marker="s", label='Setosa')
plt.scatter(classB['sepal_length'], classB['sepal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph1.png')
plt.show()


fig = plt.figure()
fig.suptitle('Sepal Length vs Petal Length',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Petal Length', fontsize=12)
plt.scatter(classA['sepal_length'], classA['petal_length'], c='g', marker="s", label='Setosa')
plt.scatter(classB['sepal_length'], classB['petal_length'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph2.png')
plt.show()


fig = plt.figure()
fig.suptitle('Sepal Length vs Petal Width',fontsize = 18)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(classA['sepal_length'], classA['petal_width'], c='g', marker="s", label='Setosa')
plt.scatter(classB['sepal_length'], classB['petal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph3.png')
plt.show()


fig = plt.figure()
fig.suptitle('Sepal Width vs Petal Length',fontsize = 18)
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Petal Length', fontsize=12)
plt.scatter(classA['sepal_width'], classA['petal_length'],c='g', marker="s", label='Setosa')
plt.scatter(classB['sepal_width'], classB['petal_length'],c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph4.png')
plt.show()

fig = plt.figure()
fig.suptitle('Sepal Width vs Petal Width',fontsize = 18)
plt.xlabel('Sepal Width', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(classA['sepal_width'], classA['petal_width'],c='g', marker="s", label='Setosa')
plt.scatter(classB['sepal_width'], classB['petal_width'],c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph5.png')
plt.show()


fig = plt.figure()
fig.suptitle('Petal Length vs Petal Width',fontsize = 18)
plt.xlabel('Petal Length', fontsize=12)
plt.ylabel('Petal Width', fontsize=12)
plt.scatter(classA['petal_length'], classA['petal_width'], c='g', marker="s", label='Setosa')
plt.scatter(classB['petal_length'], classB['petal_width'], c='r', marker="o", label='Versicolor')
plt.legend(loc='upper left')
plt.savefig('graph6.png')
plt.show()
