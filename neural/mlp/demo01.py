# 导入必要的库
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)
print(y)

# 数据预处理
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建神经网络模型并训练
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=50000)

# 训练模型
mlp.fit(X_train, y_train)
    
# 预测模型
y_pred = mlp.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")