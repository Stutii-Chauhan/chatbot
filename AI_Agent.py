import sqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
import numpy as np
from scipy import stats

# ---- Gemini setup ----
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini LLM failed: {e}"

def generate_gemini_sql(user_query):
    prompt = f"""
You are an expert SQL query generator.

Here is the table schema:
Table Name: products
Columns:
- Sales (integer)
- Quarter (text)
- Vertical (text)
- Region (text)
- Profit (integer)

Instructions:
- ONLY generate a SQL query based on the user's request.
- DO NOT solve manually.
- DO NOT perform any calculation.
- Just output a valid SQL query.
- Treat all string comparisons as CASE-INSENSITIVE
- Use UPPER(column) for text fields and compare with uppercase values.
- Always return results with columns in the following order (if used): Quarter, Region, Vertical, Sales, Profit.
- No explanation. No reasoning steps.

User request: {user_query}

SQL Query:
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini LLM failed: {e}"

# ---- Streamlit App UI ----

st.set_page_config(page_title="AI Agent", layout="wide")
st.title("Auto Agent")

# 1. Connect to the SQLite database and load the products table
conn = sqlite3.connect('mydatabase.db')
df = pd.read_sql_query("SELECT * FROM products", conn)
conn.close()

# 2. Show data preview
st.subheader("Data Preview")
st.dataframe(df.head(50))
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# 3. Ask a question
st.subheader("Ask a Question About Your Data")
user_question = st.text_input("What do you want to know?")

if user_question:
    q = user_question.lower()

    # Handle simple missing value questions separately (optional block)
    if "missing" in q:
        if "which column" in q and ("most" in q or "maximum" in q):
            missing_per_column = df.isna().sum()
            most_missing_col = missing_per_column.idxmax()
            count = missing_per_column.max()
            st.success(f"Column with the most missing values is '{most_missing_col}' with {count} missing entries.")
        elif "per column" in q or "column wise" in q or "each column" in q:
            missing_per_column = df.isna().sum()
            st.write("### Missing Values per Column")
            st.dataframe(missing_per_column[missing_per_column > 0])
        else:
            total_missing = df.isna().sum().sum()
            st.success(f"Total missing values in the dataset: {total_missing}")

    else:
        # Use Gemini to generate SQL query
        with st.spinner("Generating SQL Query..."):
            try:
                sql_query = generate_gemini_sql(user_question)
                st.code(sql_query, language='sql')

                # Optional: Execute the generated SQL query and show result
                execute_query = st.checkbox("Run this query on the database")

                if execute_query:
                    conn = sqlite3.connect('mydatabase.db')
                    try:
                        clean_query = sql_query.strip().strip("```").replace("sql", "").strip()
                        result_df = pd.read_sql_query(clean_query, conn)
                        st.success("Query executed successfully!")
                        st.dataframe(result_df)
                        if result_df.shape[0] == 1 and result_df.shape[1] == 2:
                            label_col = result_df.columns[0]
                            value_col = result_df.columns[1]
                            label = result_df.iloc[0, 0]
                            value = result_df.iloc[0, 1]

                            if pd.notna(value):
                                summary = f"The {value_col} for {label_col} '{label}' is **{int(value):,}**."
                                st.markdown(summary)
                            else:
                                st.info("No matching data found for this query.")

                    except Exception as query_error:
                        st.error(f"SQL Execution Failed: {query_error}")
                    conn.close()

            except Exception as e:
                st.error(f"Something went wrong with Gemini: {e}")
