import pandas as pd
import quandl,math,datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from statistics import mean
import matplotlib.pyplot as plt
df= quandl.get('WIKI/GOOGL',authtoken='YHDy9TSydxBKsufvxYGv')

df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

#myframe= pd.DataFrame({ "name":["venkat","prani"],"age":[21,21],"weight":[70,65]})
#print(df.head())
#print(myframe.head())

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0
df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col= 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out= int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
#print(df.head)

#X=np.array(df.drop(['label'] ,1) )
#X=preprocessing.scale(X)
#X_start=X[:-forecast_out]
#X_lately=X[-forecast_out:]
X=np.array(df['Adj. Close'])
X=preprocessing.scale(X)
X_single=X[:-forecast_out]
X_lately=X[-forecast_out:]
df.dropna(inplace=True)
Y=np.array(df['label'])
Y_lately=Y[-forecast_out:]
print(len(X_single),len(Y_lately))

#print(len(Y))
#print(len(X_start))
#y=np.array(df['label'])

X_train, X_test, Y_train, Y_test =cross_validation.train_test_split(X_single,Y, test_size=0.2)
print(len(X_train),len(Y_train))
def fit(xtrain,ytrain):
    xs=X_train
    ys=Y_train
    return xs,ys

xs,ys=fit(X_train,Y_train)
print np.mean(xs)
print len(xs)
print len(ys)

def linear_regression(xs,ys):
    first = np.mean(xs) * np.mean(ys)
    second= np.mean(xs * ys)
    third= np.mean(xs) * np.mean(xs)
    fourth = np.mean(xs * xs)

    m= ((first - second)) / ((third - fourth))
    b= mean(ys) - m * np.mean(xs) 


    return m,b

m,b=linear_regression(xs,ys)

for x in X_lately:
    predict_y=(m*x)+b
    np.append(Y_lately,predict_y)
#print(X_single)
print(Y_lately)

plt.scatter(xs,ys)

best_fit_line= [(m * x)+b for x in xs] 

plt.plot(xs, best_fit_line)
plt.show()




    
'''clf=LinearRegression() 
clf.fit(X_train,Y_train)
#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf,f)
#pickle_in= open('linearregression.pickle','rb')
#clf = pickle.load(pickle_in)
accuracy= clf.score(X_test,Y_test)
forecast_set= clf.predict(X_lately)
print(forecast_set,accuracy)
#print(accuracy)
last_date= df.iloc[-1].name
last_unix = last_date.timestamp()
one_day= 86400
next_unix = last_unix + one_day
df['Forecast']=np.nan
#print(last_date,"\n",last_unix, next_unix)
next_date = datetime.datetime.fromtimestamp(next_unix)
actual_date= datetime.datetime.fromtimestamp(last_unix)
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('price')
plt.show()'''


