import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def train_isolation_forest():
    file_path = r'C:\Users\Admin\Desktop\fitterlyf_data_science\creditcard_test.xlsx'  # Update with your local path if needed
    transactions = pd.read_excel(file_path)
    non_numeric_columns = transactions.select_dtypes(exclude=[float, int]).columns
    transactions = transactions.drop(columns=non_numeric_columns, errors='ignore')
    
    if 'Class' in transactions.columns:
        X = transactions.drop(columns=['Class'])
    else:
        st.warning("You can upload the file now")
        X = transactions.copy()

    scaler = StandardScaler()
    transactions[['Amount', 'Time']] = scaler.fit_transform(transactions[['Amount', 'Time']])
    
    # Train the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.0017, random_state=42)
    iso_forest.fit(X)
    return iso_forest, scaler, non_numeric_columns

# Streamlit app
st.title("Credit Card Fraud Detection")

st.write("""
Upload your dataset
Upload a new set of credit card transactions to detect fraudulent activities.
""")

# Train the model once
iso_forest, scaler, non_numeric_columns = train_isolation_forest()

uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file is not None:
    try:
        new_data = pd.read_excel(uploaded_file)

        # Preprocess the new data
        if 'Amount' in new_data.columns and 'Time' in new_data.columns:
            new_data = new_data.drop(columns=non_numeric_columns, errors='ignore')
            new_data[['Amount', 'Time']] = scaler.transform(new_data[['Amount', 'Time']])

            # Detect anomalies
            y_pred = iso_forest.predict(new_data)
            new_data['Anomaly'] = y_pred
            frauds = new_data[new_data['Anomaly'] == -1]

            st.write(f"### Number of detected fraudulent transactions: {frauds.shape[0]}")
            st.write(frauds)

            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_transformed = pca.fit_transform(new_data.drop(columns=['Anomaly'], errors='ignore'))
            new_data['PCA1'] = pca_transformed[:, 0]
            new_data['PCA2'] = pca_transformed[:, 1]

            # Visualization
            st.write("### PCA Visualization of Detected Anomalies")
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PCA1', y='PCA2', hue='Anomaly', data=new_data, palette='coolwarm', alpha=0.6)
            plt.title('Anomaly Detection Results (Isolation Forest)')
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            st.pyplot(plt)
        else:
            st.error("The uploaded file does not contain the required columns 'Amount' and 'Time'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a dataset to proceed.")
