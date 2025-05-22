from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier

from train import X_train, y_train

file_url= "iris_data/iris.csv"

columns =['x1','x2','x3','x4','y']

df = pd.reade_csv(file_url, columns=columns)

X = df.drop('y', axis=1)
y = df['y']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)


