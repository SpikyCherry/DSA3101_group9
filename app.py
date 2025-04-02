import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import pandas as pd

def display_notebook_content(notebook_path):
    """Displays the content of an IPython Notebook in Streamlit."""

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(notebook)

        st.components.v1.html(body, height=800, scrolling=True)

    except FileNotFoundError:
        st.error(f"Notebook not found: {notebook_path}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Streamlit App ---

st.title("Notebook and DataFrame Display")

st.header("DataFrame Used in Notebook")

try:
    # Attempt to load the DataFrame (adjust path and loading method as needed)
    # This assumes that the notebook saved the dataframe to a csv called 'data.csv'
    df = pd.read_csv("data/processed/bank_customers_train_encoded.csv") #example path.
    st.dataframe(df.head(10))  # Display the first 10 rows

except FileNotFoundError:
    st.warning("DataFrame CSV file not found. Please ensure the notebook saves the DataFrame to 'data/processed/bank_customers_train_encoded.csv'.")
except Exception as e:
    st.error(f"An error occurred while loading the DataFrame: {e}")

st.header("Notebook Content")
display_notebook_content("question_4/A4_KPI_Analysis.ipynb") #Ensure this path is correct.
