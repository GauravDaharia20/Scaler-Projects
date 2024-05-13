import streamlit as st 
import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
st.title("University Admission Prediction")
st.subheader("This Website is backed with ML model which will predict the chances of getting admission into any University :100: :ok_hand:")

#  ['const' -> 0.0,
#  'GRE Score'  -> (340, 290),
#  'TOEFL Score' -> (120, 92),
#  'University Rating' ->  {1, 2, 3, 4, 5} -> radio ,
#  'SOP' -> (5.0, 1.0) -> slider,
#  'LOR' -> (5.0, 1.0) -> slider,
#  'Research' -> (1,0) -> radio]

with open("LR_Model.pkl", "rb") as f:
    model = pickle.load(f)

c1,c2,c3 = st.columns(3)
with c1:
    Research = st.radio("Select If you belong to research background :computer:",["Yes","No"])
with c2:
    University = st.radio("Select University Rating :classical_building:",[1,2,3,4,5])
with c3:
    gre_Score = st.slider("GRE Score :mortar_board:",260,340,step=1)
c4,c5,c6 = st.columns(3)
with c4:
    TOEFL_Score = st.slider("TOEFL Score :dart:",0,120,step=1)
with c5:
    LOR = st.slider("Letter of Recommendation",1.0,5.0,step=0.5)
with c6:
    SOP = st.slider("Statement of Purpose ",1.0,5.0,step=0.5)

encode_dict ={"Yes":1,"No":0}
def model_pred(Research,University,gre_Score,TOEFL_Score,LOR,SOP):
    research = encode_dict[Research]
    data =np.array([0.0,gre_Score,TOEFL_Score,University,SOP,LOR,research])
    data = StandardScaler().fit_transform(data.reshape(-1,1))
    #st.write(data.T)
    percentage = model.predict(data.T)
    return (percentage)

admission_percentage = model_pred(Research,University,gre_Score,TOEFL_Score,LOR,SOP)
text = st.text_area("Percentage of Getting Admission :admission_tickets:",str(round(admission_percentage[0]*100,2))+" %")
