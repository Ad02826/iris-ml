import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

# 数据集路径
file_path = 'iris_data/iris.csv'

# 定义列名
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

# 读取数据
df = pd.read_csv(file_path, header=None, names=columns)

# 特征和标签
X = df.drop('Species', axis=1)
y = df['Species']

# 分割数据集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建KNN模型
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 进行预测
y_pred = knn.predict(X_test)

# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)

print(f"模型准确率: {accuracy:.2f}")

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique()
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=True)