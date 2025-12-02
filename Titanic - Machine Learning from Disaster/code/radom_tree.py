# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 加载数据
print("Loading data...")
train_data = pd.read_csv(r'/Titanic - Machine Learning from Disaster/train.csv')
test_data = pd.read_csv(r'/Titanic - Machine Learning from Disaster/test.csv')

# 显示数据基本信息
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# 准备目标变量
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

