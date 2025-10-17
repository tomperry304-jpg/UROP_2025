# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:31:07 2025

@author: tompe
"""
import pandas as pd

file_path = r"C:\Users\tompe\OneDrive - Imperial College London\Year 2\UROP\Project\WFD_SW_Classification_Status_and_Objectives_Cycle2_v4.xlsx"

sheets_dict = pd.read_excel(file_path, sheet_name=None)

# Combine all sheets into a single DataFrame with a 'year' column
df_list = []

for sheet_name, df in sheets_dict.items():
    df = df.copy()
    df["year"] = sheet_name  # tag each row with the sheet name (assumed to be the year)
    df_list.append(df)

# Combine into one big DataFrame
full_df = pd.concat(df_list, ignore_index=True)




 

X = full_df[["OV_CLASS", "ECO_CLASS", "MMA_CLASS", "year","MOR_CLASS"]]  # include year
X = X.iloc[1:]
X = X.dropna()


from sklearn.preprocessing import LabelEncoder

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
        
x=X.iloc[:,0:-1]
y=X.iloc[:,-1]   
    
    
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x,y)

y_pred = model.predict(x)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("R² Score:", r2_score(y, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y, y_pred))
print("Mean Squared Error:", mean_squared_error(y, y_pred))

import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = x.columns

# Plot
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.show()

keys = {x: y for x,y in zip(full_df.columns, df.iloc[0])}

#translate back to catagorys - look at random forrest clasifications
#mirror % distribution
#cross validation
#loop over combinations of dependent and indipendent variables
#compeare effectivness of combinations

from sklearn.metrics import accuracy_score

scores = []

for iteration in range (10):
  x_ = x.sample(int(x.shape[0] * 0.9))
  y_ = y.loc[x_.index]
  x_test = x.loc[~x.index.isin(x_.index)]
  y_test = y.loc[x_test.index]
    
    
  model.fit(x_,y_)
  y_pred = model.predict(x_test)
  
  acc = accuracy_score(y_test, y_pred)
  scores.append(acc)