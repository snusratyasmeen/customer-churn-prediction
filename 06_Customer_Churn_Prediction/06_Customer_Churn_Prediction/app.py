import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.title("📊 Customer Churn Prediction")
df=pd.read_csv("data/customers.csv"); m=RandomForestClassifier(n_estimators=100,random_state=42).fit(df[["tenure","monthly_charges","support_calls"]],df.churn)
t=st.slider("Tenure",1,72,24); c=st.slider("Monthly Charges",10.0,150.0,60.0); s=st.slider("Support Calls",0,15,2)
if st.button("Predict"): st.warning("Likely to churn" if m.predict([[t,c,s]])[0] else "Likely to stay")
st.dataframe(df)