import streamlit as st
import xgboost as xgb
import pandas as pd
from pathlib import Path

model_dir=(Path(__file__).resolve().parent.parent/'work'/'models')
@st.cache_resource
def load_model():
    model_loc=model_dir/'xgb_weighted.json'
    booster=xgb.Booster()
    booster.load_model(model_loc)
    return booster

@st.cache_data
def load_data():
    test_metadata_loc=model_dir/'test_metadata.parquet'
    X_test_loc=model_dir/'X_test.parquet'
    meta_data=pd.read_parquet(test_metadata_loc)
    X_test=pd.read_parquet(X_test_loc)
    return meta_data,X_test

booster=load_model()
meta_data, X_test=load_data()
dmatrix=xgb.DMatrix(X_test)
prob=booster.predict(dmatrix)[:,0]
prob_ser=pd.Series(prob,index=X_test.index)
meta_data=meta_data.join(prob_ser.rename('probability'))

# Selectbox
all_clients=meta_data
unique_client=meta_data['client_hash_id'].unique().tolist()
selected=st.selectbox("Clients",options=['All Clients'] + unique_client)
if selected=='All Clients':
    filtered=all_clients
else:
    filtered=all_clients[all_clients['client_hash_id']==selected]

# Slider
meta_data['probability']=meta_data['probability'].astype(float)
slid=st.slider('Select Probability Range',min_value=meta_data['probability'].min(), max_value=meta_data['probability'].max(), step=0.01,value=meta_data['probability'].quantile(0.90))
filtered=filtered[filtered['probability']>=slid]
filtered=filtered.sort_values(by='probability',ascending=False)
filtered=filtered.rename(columns={'probability':'Decline Risk (%)'})
filtered['Decline Risk (%)']=filtered['Decline Risk (%)']*100
st.caption(f"filtering {filtered.shape[0]:,} out of {meta_data.shape[0]:,} ")
st.dataframe(filtered,column_config={
    "Decline Risk (%)":st.column_config.NumberColumn("Decline Risk (%)",format="%.2f%%",)
})
