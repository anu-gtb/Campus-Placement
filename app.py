# Import libraries
import streamlit as st 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load Standardization parameters
with open('scaler.pkl','rb') as f:
   scaler=pickle.load(f)

# Load the model
with open('model.pkl','rb') as file:
   model=pickle.load(file)

# Initialize Streamlit app
st.set_page_config('Campus Placement')

st.title('Placement Prediction')

subheader_text = "Fill the following details :"
style = f"<p style='font-size:18px;font-family:serif'> {subheader_text} </p>"
st.markdown(style, unsafe_allow_html=True)

inp1=st.number_input('HSC Percentage')
inp2=st.number_input('Employability Test Percentage')
inp3=st.selectbox('Board Of Secondary Education other than Central?',['Yes','No'])
inp4=st.selectbox('Stream Of Higher Secondary Education other than Science?',['Yes','No'])
inp5=st.selectbox('Undergraduate Degree type other than Science & Technology?',['Yes','No'])
inp6=st.selectbox('Have any Work Experience?',['Yes','No'])
inp7=st.selectbox('Specialization in Marketing & HR',['Yes','No'])

submit=st.button('SUBMIT')

# Store inputs inside a dataframe
df=pd.DataFrame()
df['hsc_p']=[inp1]
df['etest_p']=[inp2]
df['ssc_b_Others']=[inp3]
df['hsc_s_Science']=[inp4]
df['degree_t_Others']=[inp5]
df['workex_Yes']=[inp6]
df['specialisation_Mkt&HR']=[inp7]

# Preprocess the inputs
df=df.replace({'Yes':1,'No':0})#OneHotEncoding

scaled_data=pd.DataFrame(scaler.transform(df))#Standardization

# If submit button clicked...
if submit:
   pred=model.predict(scaled_data)#Prediction
   if pred[0]==1:
      st.subheader('Placed')
   else:
      st.subheader('Not Placed')
   
