def generate_gemini_insight(df_sample, chart_type, x_col=None, y_col=None):
    prompt = f"""
You are an expert SQL generator.

Given the following table schema:
Table Name: products
Columns:
- Sales (integer)
- Quarter (text)
- Vertical (text)
- Region (text)
- Profit (integer)

Generate a correct SQL query for the user's request.
Do not explain anything. Just output the SQL query.

Data Sample:
{df_sample.to_csv(index=False)}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini LLM failed: {e}"


# with right_col:
#     if "df" in st.session_state:
#         df = st.session_state.df
#         numeric_cols = df.select_dtypes(include='number').columns.tolist()
#         all_cols = df.columns.tolist()

#         st.subheader("Create Your Own Chart")

#         chart_type = st.selectbox("Choose chart type", [
#             "Bar", "Column", "Pie", "Histogram",
#             "Line", "Scatter", "Box",
#             "Scatter with Regression", "Trendline",
#             "Correlation Heatmap"
#         ])

#         x_col = y_col = None
#         fig = None

#         # Axis selectors
#         if chart_type in ["Bar", "Column", "Line", "Scatter", "Box", "Scatter with Regression", "Trendline", "Histogram"]:
#             x_col = st.selectbox("Select X-axis", all_cols)

#         if chart_type in ["Line", "Scatter", "Box", "Scatter with Regression", "Trendline"]:
#             y_options = [col for col in numeric_cols if col != x_col]
#             if y_options:
#                 y_col = st.selectbox("Select Y-axis", y_options)
#             else:
#                 y_col = None
#                 st.warning("No available numeric column for Y-axis.")

#         if chart_type == "Pie":
#             x_col = st.selectbox("Select category column for pie chart", all_cols)

#         try:
#             chart_df = pd.DataFrame()
#             if chart_type in ["Bar", "Column"] and x_col:
#                 bar_mode = st.radio("How do you want to build this chart?", ["Auto Count", "Custom X and Y"], horizontal=True)

#                 if bar_mode == "Auto Count":
#                     value_counts = df[x_col].dropna().value_counts()
#                     chart_df = pd.DataFrame({x_col: value_counts.index, "Count": value_counts.values})
#                     fig = px.bar(
#                         x=chart_df[x_col] if chart_type == "Column" else chart_df["Count"],
#                         y=chart_df["Count"] if chart_type == "Column" else chart_df[x_col],
#                         orientation='v' if chart_type == "Column" else 'h',
#                         labels={"x": x_col, "y": "Count"} if chart_type == "Column" else {"y": x_col, "x": "Count"}
#                     )

#                 elif bar_mode == "Custom X and Y":
#                     y_options = [col for col in numeric_cols if col != x_col]
#                     if y_options:
#                         y_col = st.selectbox("Select Y-axis (numeric)", y_options)
#                         agg_method = st.radio("Aggregation method:", ["Sum", "Mean", "Median"], horizontal=True)
#                         agg_func = {
#                             "Sum": "sum",
#                             "Mean": "mean",
#                             "Median": "median"
#                         }[agg_method]
#                         if not (pd.api.types.is_numeric_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(df[x_col])):
#                             chart_df = df[[x_col, y_col]].dropna().groupby(x_col)[y_col].agg(agg_func).reset_index()
#                         else:
#                             chart_df = df[[x_col, y_col]].dropna()
#                         fig = px.bar(
#                             chart_df,
#                             x=x_col if chart_type == "Column" else y_col,
#                             y=y_col if chart_type == "Column" else x_col,
#                             orientation='v' if chart_type == "Column" else 'h'
#                         )
#                     else:
#                         st.warning("No valid numeric column available for Y-axis.")

#             elif chart_type == "Histogram" and x_col:
#                 chart_df = df[[x_col]].dropna()
#                 fig = px.histogram(chart_df, x=x_col)

#             elif chart_type == "Pie" and x_col:
#                 pie_vals = df[x_col].dropna().value_counts()
#                 chart_df = pd.DataFrame({x_col: pie_vals.index, "Count": pie_vals.values})
#                 fig = px.pie(names=pie_vals.index, values=pie_vals.values)

#             elif chart_type == "Line" and x_col and y_col:
#                 agg_method = st.radio("Aggregation method:", ["Sum", "Mean", "Median"], horizontal=True)
#                 agg_func = {
#                     "Sum": "sum",
#                     "Mean": "mean",
#                     "Median": "median"
#                 }[agg_method]
#                 if not (pd.api.types.is_numeric_dtype(df[x_col]) or pd.api.types.is_datetime64_any_dtype(df[x_col])):
#                     chart_df = df[[x_col, y_col]].dropna().groupby(x_col)[y_col].agg(agg_func).reset_index()
#                 else:
#                     chart_df = df[[x_col, y_col]].dropna()
#                 fig = px.line(chart_df, x=x_col, y=y_col, markers=True, text=chart_df[y_col].round(2))
#                 fig.update_traces(textposition="top center")

#             elif chart_type == "Scatter" and x_col and y_col:
#                 chart_df = df[[x_col, y_col]].dropna()
#                 fig = px.scatter(chart_df, x=x_col, y=y_col)

#             elif chart_type == "Box" and x_col and y_col:
#                 chart_df = df[[x_col, y_col]].dropna()
#                 fig = px.box(chart_df, x=x_col, y=y_col)

#             elif chart_type == "Scatter with Regression" and x_col and y_col:
#                 chart_df = df[[x_col, y_col]].dropna()
#                 fig = px.scatter(chart_df, x=x_col, y=y_col, trendline="ols")

#             elif chart_type == "Trendline" and x_col and y_col:
#                 chart_df = df[[x_col, y_col]].dropna()
#                 fig = px.scatter(chart_df, x=x_col, y=y_col, trendline="lowess")

#             elif chart_type == "Correlation Heatmap" and numeric_cols:
#                 chart_df = df[numeric_cols].corr()
#                 fig = px.imshow(chart_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')

#             if fig:
#                 st.plotly_chart(fig, use_container_width=True)

#                 # Show Gemini insight below the chart
#                 if chart_df is not None and not chart_df.empty:
#                     with st.spinner("Buzz is analyzing the chart..."):
#                         insight = generate_gemini_insight(chart_df, chart_type, x_col, y_col)
#                         insight_part = insight
#                         recommendation_part = ""
#                         if "Recommendations:" in insight:
#                             parts = insight.split("Recommendations:")
#                             insight_part = parts[0].strip()
#                             recommendation_part = parts[1].strip()
#                         st.markdown(f"""
#                             <div style="background-color:#f1f5ff; padding: 20px; border-radius: 10px;">
#                                 <h4 style="margin-bottom: 10px;">🤖 <strong>Buzz's Analysis</strong></h4>
#                                 <p style="font-size: 16px; line-height: 1.6;">
#                                     <strong>Insights:</strong> {insight_part} <br><br>
#                                     <strong>Recommendations:</strong> {recommendation_part}
#                                 </p>
#                             </div>
#                         """, unsafe_allow_html=True)

#             elif chart_type not in ["Correlation Heatmap"]:
#                 st.info("Please select appropriate columns to generate the chart.")

#         except Exception as e:
#             st.error(f"Error generating chart: {e}")
