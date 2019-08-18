print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
import random

rng = np.random.RandomState(42)

# Generate train data
PATH = "/Users/yangxuesen/Downloads/"
data = pd.read_csv(PATH + "ionosphere_data_kaggle.csv")
size_low=data.shape[0]
size_colon=data.shape[1]
d1=random.randint(0,size_colon-1)
d2=random.randint(0,size_colon-1)
X_train=data.iloc[1:200,[d1,d2]]
# Generate some regular novel observations
X_test=data.iloc[200:-1,[d1,d2]]

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination=0.05, n_jobs=-1, max_features=0.9, verbose=0) # contamination='auto'
clf.fit(X_train)
# print(clf.estimators_) # 预测器
# print(clf.estimators_samples_) ＃ 每个预测器的样本分布
y_pred_train = clf.predict(X_train) # 预估值－1表示异常 1表示正常
y_pred_test = clf.predict(X_test) # 预估值－1表示异常 1表示正常
score=clf.score_samples(X_train) # 检测值 越低越不正常 －1到0之间
y=clf.decision_function(X_train) # 检测值，小于0表示异常 －1到1之间

def pro_y(y):
    len_y=len(y)
    num=0
    for i in y:
        if i==-1:
            num=num+1
        else:
            num=num
    return num/len_y

def unique_index(L,f):
   return [i for (i,v) in enumerate(L) if v==f] # L表示列表， i表示索引值，v表示values，f表示要查找的元素

print('训练集的异常率:', pro_y(y_pred_train))
print('验证集的异常率:',pro_y(y_pred_test))

out_train=unique_index(y_pred_train,-1)
out_test=unique_index(y_pred_test,-1)
in_train=unique_index(y_pred_train,1)
in_test=unique_index(y_pred_test,1)


# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(X_train.iloc[in_train, 0], X_train.iloc[in_train, 1], c='white',
                 s=10, edgecolor='k')
b2 = plt.scatter(X_test.iloc[in_test, 0], X_test.iloc[in_test, 1], c='green',
                 s=10, edgecolor='k')
c1 = plt.scatter(X_train.iloc[out_train, 0], X_train.iloc[out_train, 1], c='yellow',
                s=10, edgecolor='k')
c2 = plt.scatter(X_test.iloc[out_test, 0], X_test.iloc[out_test, 1], c='red',
                s=10, edgecolor='k')
plt.axis('tight')
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
plt.legend([b1, b2, c1, c2],
           ["inline training observations",
            "inline testing observations",
            "outline training observations",
            "outline testing observations"],
           loc="upper left")
plt.show()
