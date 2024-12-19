import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats import gmean, hmean

def load_data(uploaded_file):
    # Load data from uploaded file
    df = pd.read_csv(uploaded_file)

    # Handle missing 'lot_size_units' using mode imputation
    mode_lot_size_units = df['lot_size_units'].mode()[0]
    df['lot_size_units'].fillna(mode_lot_size_units, inplace=True)

    # Convert 'lot_size' to sqft if it's in acres
    acre_to_sqft = 43560
    df.loc[df['lot_size_units'] == 'acre', 'lot_size'] *= acre_to_sqft
    df['lot_size_units'] = 'sqft'  # After conversion, all units are now sqft

    # Handle missing values in 'lot_size' using mean imputation
    mean_lot_size = df['lot_size'].mean()
    df['lot_size'].fillna(mean_lot_size, inplace=True)

    return df

# Quantiles
def calculate_quantiles(data, column):
    column_data = data[column].dropna()
    quartiles = column_data.quantile([0.25, 0.5, 0.75])
    deciles = column_data.quantile([i / 10 for i in range(1, 10)])
    percentiles = column_data.quantile([i / 100 for i in range(1, 100)])
    return quartiles, deciles, percentiles

# Measures of Distribution
def calculate_distribution(data, column):
    column_data = data[column].dropna()
    kurt = kurtosis(column_data)
    skewness = skew(column_data)
    kurt_category = "Mesokurtic"
    if kurt > 3:
        kurt_category = "Leptokurtic"
    elif kurt < 3:
        kurt_category = "Platykurtic"
    skew_category = "Positive Skew" if skewness > 0 else "Negative Skew"
    return {
        "Kurtosis": kurt,
        "Kurtosis Category": kurt_category,
        "Skewness": skewness,
        "Skewness Category": skew_category
    }

# Pearson’s and Bowley’s Skewness
def calculate_skewness_measures(data, column):
    column_data = data[column].dropna()
    mean = column_data.mean()
    mode = column_data.mode().iloc[0] if not column_data.mode().empty else None
    median = column_data.median()

    pearson_skewness = 3 * (mean - median) / column_data.std()
    bowley_skewness = (column_data.quantile(0.75) + column_data.quantile(0.25) - 2 * median) / (
            column_data.quantile(0.75) - column_data.quantile(0.25)
    )
    return {
        "Pearson's Skewness": pearson_skewness,
        "Bowley's Skewness": bowley_skewness
    }

# Moments and Related Measures
def calculate_moments(data, column):
    column_data = data[column].dropna()
    mean = column_data.mean()
    moments_about_origin = [np.mean(column_data ** i) for i in range(1, 5)]
    moments_about_mean = [np.mean((column_data - mean) ** i) for i in range(1, 5)]
    return moments_about_origin, moments_about_mean

# Streamlit App
def main():
    st.set_page_config(page_title="Data Analysis Dashboard", page_icon=":bar_chart:")

    # Title Page
    if "show_main_page" not in st.session_state:
        st.session_state.show_main_page = False
    
    if not st.session_state.show_main_page:
        st.title("Data Analysis Dashboard")
        st.subheader("BSAI-F-23-A | Probability & Statistics")
        st.text("Name: Muhammad Taha Mansoor")
        st.text("Reg No: 231234")  # Replace with actual registration number
        st.markdown(
            """
            ### Project Overview
            This project is designed as part of the Probability & Statistics course. The goal is to provide an interactive
            dashboard for analyzing datasets using statistical measures and visualizations. Explore quantiles, moments,
            skewness, and more, with seamless integration of graphical and descriptive insights.
            """
        )

        # Add a button to navigate to the dataset description and upload page
        if st.button("Proceed to Dataset Description"):
            st.session_state.show_main_page = True
            st.experimental_rerun()
    else:
        # Dataset Description and Upload Page
        if "uploaded_file" not in st.session_state:
            st.title("Dataset Description")
            st.markdown(
                """
                ### Required Dataset Format
                - **Columns**: 
                  - `lot_size`: Numeric values representing lot size.
                  - `lot_size_units`: Categorical values (`sqft` or `acre`).
                - **Units**:
                  - Lot size must be in square feet (`sqft`). If given in acres, it will be converted automatically.
                - **Cleaning**:
                  - Ensure missing values are minimal or cleaned appropriately.

                Please upload a cleaned dataset that adheres to this format.
                """
            )

            st.session_state.uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv"])

            if st.session_state.uploaded_file is not None:
                st.success("Dataset uploaded successfully!")
                st.session_state.data = load_data(st.session_state.uploaded_file)
                st.experimental_rerun()
        else:
            # Main App
            df = st.session_state.data

            st.sidebar.title("Dashboard")
            menu_options = [
                "Graphical Representation", "Descriptive Measures", "Quantiles", 
                "Distribution Measures", "Skewness Measures", "Moments"
            ]
            selected_option = st.sidebar.radio("Select a Section:", menu_options)

            # Graphical Representation
            if selected_option == "Graphical Representation":
                st.title("Graphical Representation")
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                st.sidebar.header("Graph Options")
                graph_type = st.sidebar.selectbox("Select Graph Type", ["Scatter Plot", "Bar Plot", "Line Plot"])
                x_column = st.sidebar.selectbox("Select X-axis", numeric_columns)
                y_column = st.sidebar.selectbox("Select Y-axis", numeric_columns)

                if graph_type == "Scatter Plot":
                    st.write(f"### Scatter Plot: {x_column} vs {y_column}")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
                    st.pyplot(fig)
                elif graph_type == "Bar Plot":
                    st.write(f"### Bar Plot: {x_column} vs {y_column}")
                    fig, ax = plt.subplots()
                    sns.barplot(x=df[x_column], y=df[y_column], ax=ax)
                    st.pyplot(fig)
                elif graph_type == "Line Plot":
                    st.write(f"### Line Plot: {x_column} vs {y_column}")
                    fig, ax = plt.subplots()
                    sns.lineplot(x=df[x_column], y=df[y_column], ax=ax)
                    st.pyplot(fig)

            # Descriptive Measures
            elif selected_option == "Descriptive Measures":
                st.title("Descriptive Measures")
                
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                selected_column = st.selectbox("Select a Column for Descriptive Statistics:", numeric_columns)
                
                if selected_column:
                    st.write(f"Descriptive Statistics for {selected_column}:")
                    st.write(f"- Mean: {df[selected_column].mean()}")
                    st.write(f"- Median: {df[selected_column].median()}")
                    st.write(f"- Standard Deviation: {df[selected_column].std()}")
                    st.write(f"- Minimum: {df[selected_column].min()}")
                    st.write(f"- Maximum: {df[selected_column].max()}")

            # Quantiles
            elif selected_option == "Quantiles":
                st.title("Quantiles")
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                selected_column = st.selectbox("Select a Column for Quantiles:", numeric_columns)
                quartiles, deciles, percentiles = calculate_quantiles(df, selected_column)
                st.write("### Quartiles", quartiles)
                st.write("### Deciles", deciles)
                st.write("### Percentiles", percentiles)

            # Distribution Measures
            elif selected_option == "Distribution Measures":
                st.title("Measures of Distribution")
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                selected_column = st.selectbox("Select a Column for Distribution:", numeric_columns)
                results = calculate_distribution(df, selected_column)
                for key, value in results.items():
                    st.write(f"- **{key}**: {value}")

            # Skewness Measures
            elif selected_option == "Skewness Measures":
                st.title("Skewness Measures")
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                selected_column = st.selectbox("Select a Column for Skewness:", numeric_columns)
                results = calculate_skewness_measures(df, selected_column)
                for key, value in results.items():
                    st.write(f"- **{key}**: {value}")

            # Moments
            elif selected_option == "Moments":
                st.title("Moments")
                numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                selected_column = st.selectbox("Select a Column for Moments:", numeric_columns)
                moments_origin, moments_mean = calculate_moments(df, selected_column)
                st.write("### Moments About Origin")
                st.write(moments_origin)
                st.write("### Moments About Mean")
                st.write(moments_mean)

if __name__ == "__main__":
    main()
