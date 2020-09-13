import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Operand_data.csv")


cdf = df[["First_operand","Second_operand","Third_operand","result"]]

x = cdf.iloc[:,:3]
y = cdf.iloc[:,-1]
print(x)

regressor = LinearRegression()

#Fitting model with training data
regressor.fit(x,y)

pickle.dump(regressor,open('model.pk1','wb'))

#model = pickle.load(open('model.pk1','rb'))

#print(model.predict([[2,2,2]]))