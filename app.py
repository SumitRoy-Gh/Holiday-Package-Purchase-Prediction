import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

@st.cache_data
def load_data(csv_path="Travel.csv"):
    df = pd.read_csv(csv_path)
    # data cleaning from your notebook
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

    # Impute missing values based on your notebook strategy
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['TypeofContact'].fillna(df['TypeofContact'].mode()[0], inplace=True)
    df['DurationOfPitch'].fillna(df['DurationOfPitch'].median(), inplace=True)
    df['NumberOfFollowups'].fillna(df['NumberOfFollowups'].mode()[0], inplace=True)
    df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mode()[0], inplace=True)
    df['NumberOfTrips'].fillna(df['NumberOfTrips'].median(), inplace=True)
    df['NumberOfChildrenVisiting'].fillna(df['NumberOfChildrenVisiting'].mode()[0], inplace=True)
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

    # feature creation
    df['TotalVisiting'] = df['NumberOfPersonVisiting'].fillna(0) + df['NumberOfChildrenVisiting'].fillna(0)
    df.drop(['CustomerID', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)
    return df

@st.cache_data
def build_pipeline(df):
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    cat_features = X.select_dtypes(include="object").columns.tolist()
    num_features = X.select_dtypes(exclude="object").columns.tolist()

    # scikit-learn 1.2+ uses sparse_output, older versions use sparse
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        OneHotEncoder(sparse_output=False, **ohe_kwargs)
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(**ohe_kwargs), cat_features),
        ("num", StandardScaler(), num_features),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    model.fit(X, y)
    return model

def get_input_data(df):
    st.sidebar.header("Fill customer values:")
    sidebar_values = {}
    # numeric columns
    num_cols = ["Age", "CityTier", "DurationOfPitch", "NumberOfFollowups",
                "PreferredPropertyStar", "NumberOfTrips", "Passport",
                "PitchSatisfactionScore", "OwnCar", "TotalVisiting", "MonthlyIncome"]
    for col in num_cols:
        min_v = int(df[col].min())
        max_v = int(df[col].max())
        default = int(df[col].median())
        sidebar_values[col] = st.sidebar.slider(col, min_value=min_v, max_value=max_v, value=default)

    # categorical columns
    cat_cols = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
    for col in cat_cols:
        opts = sorted(df[col].dropna().unique())
        sidebar_values[col] = st.sidebar.selectbox(col, opts)

    return pd.DataFrame([sidebar_values])

def main():
    st.title("Holiday Package Purchase Prediction (Random Forest)")
    st.write("Enter customer profile details in sidebar and click Predict.")

    df = load_data()

    if os.path.exists("rf_travel_model.pkl"):
        model = joblib.load("rf_travel_model.pkl")
    else:
        model = build_pipeline(df)
        joblib.dump(model, "rf_travel_model.pkl")

    st.subheader("Dataset preview")
    st.write(df.head(5))

    input_df = get_input_data(df)
    st.subheader("Customer input")
    st.write(input_df)

    if st.button("Predict Product Purchase"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.markdown(f"**Prediction:** {'Will Purchase (1)' if pred==1 else 'Will not Purchase (0)'}")
        st.markdown(f"**Probability of purchase:** {proba:.2%}")

    st.subheader("Model info")
    st.write("Random Forest classifier trained on full dataset with ColumnTransformer pipeline.")

if __name__ == "__main__":
    main()   