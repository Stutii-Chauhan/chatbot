import sqlite3
import streamlit as st
import pandas as pd
import google.generativeai as genai
import re
import numpy as np
from scipy import stats

# ---- Gemini setup ----
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash-lite")

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini LLM failed: {e}"

def generate_gemini_sql(user_query):
    TABLE_SCHEMAS = {
        "product_price_cleaned_output": ["product_url", "product_name", "product_price", "product_code", "brand"],
        "All - Product Count_output": ["brand", "10k–15k", "15k–25k", "25k–40k", "40k+", "<10k"],
        "All - SKU Count_output": ["brand", "10k–15k", "15k–25k", "25k–40k", "40k+", "<10k"],
        "Top 1000 - Product Count_output": ["brand", "Top 1000 Product Count"],
        "Top 1000 - SKU Count_output": ["brand", "Top 1000 SKU Count"],
        "Men - Product Count_output": ["brand", "Men - Product Count"],
        "Men - SKU Count_output": ["brand", "Men - SKU Count"],
        "Women - Product Count_output": ["brand", "Women - Product Count"],
        "Women - SKU Count_output": ["brand", "Women - SKU Count"],
        "Best Rank_All_output": ["brand", "Best Rank (First Appearance)"],
        "men_price_range_top100_output": ["brand", "10k–15k", "15k–25k", "25k–40k", "40k+", "total"],
        "women_price_range_top100_output": ["brand", "10k–15k", "15k–25k", "25k–40k", "40k+", "total"],
        "Final_Watch_Dataset_Men_output": ["URL", "Brand", "Product Name", "Model Number", "Price", "Ratings", "Discount", "Band Colour", "Band Material", "Band Width", "Case Diameter", "Case Material", "Case Thickness", "Dial Colour", "Crystal Material", "Case Shape", "Movement", "Water Resistance Depth", "Special Features", "ImageURL"],
        "Final_Watch_Dataset_Women_output": ["URL", "Brand", "Product Name", "Model Number", "Price", "Ratings", "Discount", "Band Colour", "Band Material", "Band Width", "Case Diameter", "Case Material", "Case Thickness", "Dial Colour", "Crystal Material", "Case Shape", "Movement", "Water Resistance Depth", "Special Features", "ImageURL"]
    }

    schema_text = "\n".join([
        f"- {table}: columns = [{', '.join(columns)}]"
        for table, columns in TABLE_SCHEMAS.items()
    ])

    prompt = f"""
You are an expert SQL generator.

Below are the available tables and their exact column names:
{schema_text}

Instructions:
- Select the most appropriate table for the user’s question.
- ONLY use the columns exactly as written for the selected table.
- DO NOT add or assume extra columns or values.
- DO NOT explain the query — just return raw SQL.
- If the question doesn’t map to any table, return exactly: INVALID_QUERY

User question: {user_query}

SQL Query:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini LLM failed: {e}"

#Chart generator

def get_chart_type_from_llm(user_query):
    prompt = f"""
- You are a data visualization expert. A user has asked a question about their dataset.
- Based on the question, decide the most appropriate chart type to visualize the SQL query result.
- Choose one of: bar, line, pie, scatter, heatmap, none
- Only return the chart type in lowercase. No explanation, no formatting.

Question: "{user_query}"
"""
    try:
        response = model.generate_content(prompt).text.strip().lower()
        return response if response in ['bar', 'line', 'pie', 'scatter', 'heatmap'] else 'none'
    except Exception as e:
        return "none"

##If a chart is needed 
def should_include_chart(user_query):
    prompt = f"""
You are a helpful analytics assistant.

Decide whether a chart would help explain this business question: "{user_query}"

Answer with: yes or no. Do not explain.
"""
    try:
        reply = model.generate_content(prompt).text.strip().lower()
        return reply.startswith("y")
    except:
        return False

# ---- Streamlit App UI ----

st.set_page_config(page_title="AI Agent", layout="wide")
st.title("Auto Agent")

# 1. Connect to the SQLite database and load the products table
@st.cache_data
def load_data():
    # Load secrets (already defined in your secrets.toml)
    DB = st.secrets["SUPABASE_DB"]
    USER = st.secrets["SUPABASE_USER"]
    PASSWORD = st.secrets["SUPABASE_PASSWORD"]
    HOST = st.secrets["SUPABASE_HOST"]
    PORT = st.secrets["SUPABASE_PORT"]
    
    # SQLAlchemy engine for Supabase
    engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")
    df = pd.read_sql_query("SELECT * FROM products", engine)
    engine.close()
    return df

df = load_data()

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
                chart_type = get_chart_type_from_llm(user_question)
                
                # 🔁 If Gemini said no chart, but user question implies one is helpful, force it
                if chart_type == "none" and should_include_chart(user_question):
                    chart_type = "bar"
                
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
                            #conn = sqlite3.connect('mydatabase.db')
                            result_df = pd.read_sql_query(clean_query, engine)
                            st.success("Query executed successfully!")
                            st.dataframe(result_df)

                            # Intelligent difference summary for 2-row comparisons
                            if result_df.shape[0] == 2 and result_df.shape[1] == 2:
                                try:
                                    cat_col, val_col = result_df.columns[0], result_df.columns[1]
                                    row1, row2 = result_df.iloc[0], result_df.iloc[1]
                                    diff = abs(int(row1[1]) - int(row2[1]))
                                    winner = row1 if row1[1] > row2[1] else row2
                                    loser = row2 if winner.equals(row1) else row1
                                    st.markdown(f"💬 **{val_col} in {winner[0]} ({int(winner[1]):,}) were greater than in {loser[0]} ({int(loser[1]):,}) by {diff:,}.**")
                                except Exception as e:
                                    st.info("Could not generate numeric insight.")
                
                            # Auto Chart Rendering (based on Gemini-inferred chart_type)
                            if chart_type != "none" and not result_df.empty:
                                st.subheader("Chart Visualization")
                
                                try:
                                    import plotly.express as px
                                    if chart_type == "bar":
                                        st.bar_chart(result_df.set_index(result_df.columns[0]))
                                    elif chart_type == "line":
                                        st.line_chart(result_df.set_index(result_df.columns[0]))
                                    elif chart_type == "pie" and result_df.shape[1] >= 2:
                                        fig = px.pie(result_df, names=result_df.columns[0], values=result_df.columns[1])
                                        st.plotly_chart(fig)
                                    elif chart_type == "scatter" and result_df.shape[1] >= 2:
                                        fig = px.scatter(result_df, x=result_df.columns[0], y=result_df.columns[1])
                                        st.plotly_chart(fig)
                                    else:
                                        st.info("Chart type suggested, but not enough data to render it.")
                                except Exception as e:
                                    st.warning(f"Chart rendering failed: {e}")

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
                                                f" The highest profit for {max_row['Vertical']} was in **{max_row['Quarter']}**, "
                                                f"**{max_row['Region']}** region with a profit of **{int(max_row['Profit']):,}**."
                                            )
                                        elif 'Region' in cols:
                                            max_row = result_df.loc[result_df['Profit'].idxmax()]
                                            st.markdown(
                                                f" The {max_row['Region']} region achieved the highest profit of **{int(max_row['Profit']):,}**."
                                            )

                                    elif 'Sales' in cols and 'Region' in cols:
                                        max_row = result_df.loc[result_df['Sales'].idxmax()]
                                        st.markdown(
                                            f" The {max_row['Region']} region had the highest sales of **{int(max_row['Sales']):,}**."
                                        )

                                    elif 'AVG(Sales)' in cols:
                                        avg_value = result_df.iloc[0, 0]
                                        st.markdown(
                                            f" The average sales is approximately **{int(avg_value):,}**."
                                        )

                                    else:
                                        # LLM fallback for small result sets
                                        if result_df.shape[0] <= 5 and result_df.shape[1] <= 3:
                                            try:
                                                preview = result_df.to_markdown(index=False)
                                                llm_prompt = f"""User asked: "{user_question}"
                                        
                                        This is the SQL query result:
                                        {preview}
                                        Instructions:
- If the result only contains a difference, do not assume directionality. Avoid guessing which is higher unless both original values are present.
- If both values are available, identify which is greater, by how much, and mention it clearly with both numbers.
- Use actual numbers in the output. Be concise, correct, and do not speculate.
- Return only a one-line insight.
                                        
                                        
                                        Write a one-line, accurate business insight that answers the question. Reflect correct directionality (e.g. which has greater profit or greater sales), use actual numbers, and avoid vague language."""
                                                llm_summary = query_gemini(llm_prompt)
                                                st.markdown(f"💬 {llm_summary}")
                                            except Exception as e:
                                                st.info(f"Could not generate a summary insight: {e}")
                                        
                                        elif result_df.shape[0] > 100:
                                            st.info("Too many rows to summarize meaningfully. Please filter your query.")
                                        else:
                                            st.info("No suitable columns found to generate a summary.")


                                except Exception as e:
                                    st.info(f"Could not generate a summary insight: {e}")

                    except Exception as query_error:
                        st.error(f"SQL Execution Failed: {query_error}")
                    finally:
                        try:
                            engine.close()
                        except:
                            pass

            except Exception as e:
                st.error(f"Something went wrong while generating the SQL: {e}")
