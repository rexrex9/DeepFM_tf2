import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def readData():

    data = pd.read_csv('data/criteo_sample.txt')
    sparse_features = ['C' + str(i) for i in range(1, 27)]  #类别型特征
    dense_features = ['I'+str(i) for i in range(1, 14)] #连续型特征

    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    #将类别型特征硬编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    labels = data['label']
    X = data[sparse_features+dense_features]

    #X=data[sparse_features]
    return np.array(X),np.array(labels)


if __name__ == '__main__':
    print(readData())
