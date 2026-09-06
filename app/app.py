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
st.write(booster.predict(dmatrix).shape)
prob=booster.predict(dmatrix)[:,0]
prob_ser=pd.Series(prob,index=X_test.index)
meta_data=meta_data.join(prob_ser.rename('probability'))

all_clients=meta_data
unique_client=meta_data['client_hash_id'].unique().tolist()
selected=st.selectbox("Clients",options=['All Clients'] + unique_client)
if selected=='All Clients':
    filtered=all_clients
else:
    filtered=all_clients[all_clients['client_hash_id']==selected]
st.dataframe(filtered)