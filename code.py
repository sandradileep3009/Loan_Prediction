import numpy as np
import pandas as pd
import pickle 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
df=pd.read_csv("loan.csv")
print("Shape: ",df.shape)
print("\n First rows: ",df.head())
print("\n Last rows: ",df.tail())
print("\n Info of data: ",df.info())

print("Null data:",df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

print(df['Loan_Status'].value_counts(normalize=True)*100)

df=df.dropna()

df.fillna(df.mean(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)
'''
plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status",data=df)
plt.title("Loan Approval Distribution")
plt.show()

print(df['Loan_Status'].value_counts(normalize=True)*100)

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),cbar=False)
plt.title("Missing Features")
plt.show()

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols].hist(figsize=(15,10),bins=20)
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Loan_Status',y='LoanAmount',data=df)
plt.title("Loan Amount vs Loan Approval")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['ApplicantIncome'],kde=True)
plt.title("Applicant Income Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Loan_Status',y='ApplicantIncome',data=df)
plt.title("Income vs Loan Approval")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Education',hue='Loan_Status',data=df)
plt.title("Education vs Loan Approval")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Credit_History',hue='Loan_Status',data=df)
plt.title("Credit History vs Loan Approval")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Property_Area',hue='Loan_Status',data=df)
plt.title("Property Area vs Loan Approval")
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='Married',hue='Loan_Status',data=df)
plt.title("Marital Status vs Loan Approval")
plt.show()

sns.pairplot(df,hue='Loan_Status')
plt.show()
'''
scaler=StandardScaler()
df.drop("Loan_ID", axis=1, inplace=True)
X = df.drop("Loan_Status", axis=1)
Y = df["Loan_Status"].map({"N":0,"Y":1})

X = pd.get_dummies(X, drop_first=True)
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

models={"Logistic Regression": LogisticRegression(),"Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),"XGB":XGBClassifier(eval_metric="logloss")}
results={}

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    results[name]=accuracy_score(y_test,y_pred)

best_model_name=max(results,key=results.get)
print("The best model: ",best_model_name)
print("The accuracy: ",results[best_model_name])

best_model=models[best_model_name]
model=best_model.fit(x_train,y_train
                     )
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

print("Accuracy :", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall   :", recall_score(y_test,y_pred))
print("F1 Score :", f1_score(y_test,y_pred))
print("ROC AUC  :", roc_auc_score(y_test,y_prob))
print(confusion_matrix(y_test,y_pred))

pickle.dump(best_model,open("loan.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))