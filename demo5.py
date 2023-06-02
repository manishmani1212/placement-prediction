import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Step 1
#Loading the dataset:

df= pd.read_csv("dataset1.csv")


#Step 2
#Data preproccessing

le=LabelEncoder()
stream=le.fit_transform(df['Stream'])
df["Stream"]=stream
x=df.pop("Stream")
df.insert(2,"Stream",x)


x=le.fit_transform(df["Gender"])
df.drop("Gender",axis=1,inplace=True)
df.insert(1,"Gender",x)

#Step 3
#Building our model

x_train,x_test,y_train,y_test=train_test_split(df[['Age','Gender','Stream','Internships','CGPA','HistoryOfBacklogs']],df.PlacedOrNot,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

def fun():
    st.header("placement prediction")
    st.info("enter all details properly!")
    Age=st.number_input("enter your age:")
    Gender=st.radio("enter your grnder",["male","female"])
    Stream=st.selectbox("enter stream",["cse","ece","mech"])
    Internships=st.number_input("enter no of intenships done yet")
    CGPA=st.number_input("enter cgpa")
    HistoryOfBacklogs=st.number_input("enter no of backlogs")
    if Gender=="male":
        Gender=1
    else:
        Gender=0
    
    if Stream=="cse":
        Stream=1
    elif Stream=="ece":
        Stream-2
    else:
        Stream=3
        
    li=[Age,Gender,Stream,Internships,CGPA,HistoryOfBacklogs]
    x=st.button("Submit")
    if x:
        output=model.predict([li])
        if output==1:
            st.success("yes placement is eligible")
        else:
            st.error("placement is not elgible")
fun()


#Step 4
#Building Streamlit app
