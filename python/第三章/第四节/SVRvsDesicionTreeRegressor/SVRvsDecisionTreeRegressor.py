import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("./data.csv")
x = df[['AT', 'V', 'AP', 'RH']]
y = df[['PE']]

sc = StandardScaler()
x_pre = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_pre, y, test_size=0.3, random_state=123)
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]

svr = SVR(gamma='auto')
svr.fit(x_train, y_train.values.ravel())
svr_predict = svr.predict(x_test)

print('='*20,'SVR','='*20)
for m in model_metrics_name:
    tmp_score = m(y_test, svr_predict)
    print(m.__name__,'=',tmp_score)

decision_tree_regressor = DecisionTreeRegressor(criterion='mse', max_depth=50)
decision_tree_regressor.fit(x_train, y_train.values.ravel())
decision_tree_regressor_predict = decision_tree_regressor.predict(x_test)
decision_tree_regressor_score = decision_tree_regressor.score(x_test, y_test)


print('='*20,'DecisionTreeRegressor','='*20)
for m in model_metrics_name:
    tmp_score = m(y_test, decision_tree_regressor_predict)
    print(m.__name__,'=',tmp_score)


print('='*20,'总结','='*20)
print('SVR的explained_variance_score（解释方差）和r2_score（判定系数）比DecisionTreeRegressor的要大，更接近于1')
print('SVR的mean_absolute_error（平均绝对误差）和mean_squared_error（均方差）比DecisionTreeRegressor的要小')
print('因此，SVR的拟合效果比DecisionTreeRegressor要好')
