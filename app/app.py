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