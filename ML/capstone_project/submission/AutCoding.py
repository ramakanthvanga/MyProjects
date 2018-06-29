from pandas import DataFrame, read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

###
data=pd.read_csv('auto_coding_extract.csv', sep=',',verbose = False,float_precision=None,dtype = str)
data=data.fillna(value='1',axis=1)
features=pd.DataFrame(data)
mask = features['VENDOR_ID'] == '462'
features = features[mask]
le_dict = {col: LabelEncoder() for col in data.columns }

#Vendor Id-  23348-100%, 462-68%, 959-31.49%, 60650-26.26%, 629-18.67%, 5628-68.52%
data=features
for col in data.columns:
  features[col]=le_dict[col].fit_transform(data[col])

coding = features['CHARGE_ACCOUNT_ID']
##Store only features by dropping final value
features = features.drop('CHARGE_ACCOUNT_ID', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, coding, test_size = 0.2, random_state = 12)
model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model_score = accuracy_score(y_test, predictions)
print("model score is: " + str(model_score))
#rowcount=data.nunique()
#index=np.arange(len(rowcount))
#plt.plot(rowcount)
#plt.xlabel('Features',fontsize=5)
#plt.ylabel('Unique count',fontsize=5)
#plt.xlabel('Features',fontsize=5)
#plt.show()