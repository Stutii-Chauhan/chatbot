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
- You ARE allowed to use arithmetic in SQL (e.g., SUM(Profit) / SUM(Sales) * 100).
- DO NOT compute the answer yourself. Just generate SQL.
- DO NOT explain the query or include any extra text.
- Treat all string comparisons as CASE-INSENSITIVE using UPPER(column).
- Always return results with columns in the following order if used: Quarter, Region, Vertical, Sales, Profit.
- If the user’s question is unrelated or unclear, reply with exactly: INVALID_QUERY

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

    # Handle simple missing value questions separately
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
        with st.spinner("Generating SQL Query..."):
            try:
                sql_query = generate_gemini_sql(user_question)
                clean_query = (
                    sql_query.strip()
                    .replace("```sql", "")
                    .replace("```", "")
                    .replace("_", "")
                    .strip()
                    .lower()
                )

                if sql_query.strip().lower().startswith("invalid_query"):
                    st.markdown(
                        """
                        <div style='
                            background-color: #e6f4ea;
                            padding: 10px 15px;
                            border-radius: 6px;
                            font-style: italic;
                            color: #1a7f37;
                        '>
                            Sorry, I didn't understand the question.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.stop()

                st.code(sql_query, language='sql')
                execute_query = st.checkbox("Run this query on the database")

                if execute_query:
                    try:
                        clean_query = sql_query.strip().strip("```").replace("sql", "").strip()

                        if "select" not in clean_query.lower():
                            st.warning("That doesn't seem like a valid question. Please rephrase your question.")
                        else:
                            conn = sqlite3.connect('mydatabase.db')
                            result_df = pd.read_sql_query(clean_query, conn)
                            st.success("Query executed successfully!")
                            st.dataframe(result_df)

                            # 1-row, 2-column summary
                            if result_df.shape[0] == 1 and result_df.shape[1] == 2:
                                label_col = result_df.columns[0]
                                value_col = result_df.columns[1]
                                label = result_df.iloc[0, 0]
                                value = result_df.iloc[0, 1]
                                if pd.notna(value):
                                    summary = f"➡️ The {value_col} for {label_col} '{label}' is **{int(value):,}**."
                                    st.markdown(summary)
                                else:
                                    st.info("No matching data found for this query.")

                            # Intelligent summaries
                            elif result_df.shape[0] > 0:
                                try:
                                    cols = result_df.columns.tolist()

                                    if 'Profit' in cols:
                                        if {'Vertical', 'Quarter', 'Region'}.issubset(cols):
                                            max_row = result_df.loc[result_df['Profit'].idxmax()]
                                            st.markdown(
                                                f"💡 The highest profit for {max_row['Vertical']} was in **{max_row['Quarter']}**, "
                                                f"**{max_row['Region']}** region with a profit of **{int(max_row['Profit']):,}**."
                                            )
                                        elif 'Region' in cols:
                                            max_row = result_df.loc[result_df['Profit'].idxmax()]
                                            st.markdown(
                                                f"💡 The {max_row['Region']} region achieved the highest profit of **{int(max_row['Profit']):,}**."
                                            )

                                    elif 'Sales' in cols and 'Region' in cols:
                                        max_row = result_df.loc[result_df['Sales'].idxmax()]
                                        st.markdown(
                                            f"💡 The {max_row['Region']} region had the highest sales of **{int(max_row['Sales']):,}**."
                                        )

                                    elif 'AVG(Sales)' in cols:
                                        avg_value = result_df.iloc[0, 0]
                                        st.markdown(
                                            f"💡 The average sales is approximately **{int(avg_value):,}**."
                                        )

                                    else:
                                        # LLM fallback for small result sets
                                        if result_df.shape[0] <= 5 and result_df.shape[1] <= 3:
                                            llm_prompt = f"""This is a table output from a SQL query:
{result_df.to_markdown(index=False)}

Write one short business-style sentence summarizing the main insight."""
                                            llm_summary = query_gemini(llm_prompt)
                                            st.markdown(f"💬 {llm_summary}")
                                        else:
                                            st.info("No suitable columns found to generate a summary.")

                                except Exception as e:
                                    st.info(f"Could not generate a summary insight: {e}")

                    except Exception as query_error:
                        st.error(f"SQL Execution Failed: {query_error}")
                    finally:
                        try:
                            conn.close()
                        except:
                            pass

            except Exception as e:
                st.error(f"Something went wrong while generating the SQL: {e}")
