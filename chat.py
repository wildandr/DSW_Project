import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import os
import sklearn

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")

# Create a Streamlit app
st.title("Telcomsel Customer Churn Dashboard (Monthly)")

# Handle file upload
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))

if fl is not None:
    # If a file is uploaded, read it
    filename = fl.name
    st.write(filename)
    if filename.endswith(".csv") or filename.endswith(".txt"):
        data = pd.read_csv(fl, encoding="ISO-8859-1")
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        data = pd.read_excel(fl)
else:
    # If no file is uploaded, try to read the file from the current directory
    filepath = os.path.join(os.getcwd(), "Telco_customer_churn_adapted_v2.xlsx")
    if os.path.exists(filepath):
        data = pd.read_excel(filepath)
    else:
        # If the file is not found, display a message and return
        st.error("The file 'Telco_customer_churn_adapted_v2.xlsx' is not found. Please upload a file.")
        st.stop()

# Buat Customer Tenure Class berdasarkan Tenure Months
data_tenure_months = data['Tenure Months']
data_tenure_months_class = []

for dtm in data_tenure_months:
    if dtm <= 6:
        data_tenure_months_class.append("New Customer")
    elif dtm <= 48:
        data_tenure_months_class.append("Regular Customer")
    else:
        data_tenure_months_class.append("Long-Term Customer")

# Tambahkan Customer Tenure Class ke dalam DataFrame
data['Customer Tenure Class'] = data_tenure_months_class

# Sidebar for user options
st.sidebar.header("Options")

# Radio button for Overview or Location Filter
selected_option = st.sidebar.radio("Select an option", ["Overview", "Overview Churn", "Filter by Location", "Prediction"])

if selected_option == "Overview":
    # Create a row for layout
    row1 = st.columns(4)

    # Row 1, Column 1
    with row1[0]:
        # Number of Customers
        st.subheader("Number of Customers")
        st.markdown(f"<h1 style='color: lightgreen; font-size: 48px; text-align: left;'>{data['Customer ID'].count()}</h1>", unsafe_allow_html=True)

    # Row 1, Column 2
    with row1[1]:
        # Number of Churned Customers
        st.subheader("Churned Customers")
        churn_count = data['Churn Label'].value_counts().get('Yes', 0)
        st.markdown(f"<h2 style='color: lightgreen; font-size: 48px; text-align: left;'>{churn_count}</h1>", unsafe_allow_html=True)

    # Row 1, Column 3
    with row1[2]:
        # Churn Rate
        st.subheader("Churn Rate")
        churn_rate = round(churn_count / data['Customer ID'].count() * 100, 2)
        st.markdown(f"<p style='color: lightgreen; font-size: 48px; text-align: left;'>{churn_rate}%</p>", unsafe_allow_html=True)

        # Row 1, Column 4
    with row1[3]:
        # Average Monthly Purchase
        st.subheader("Average Purchase")
        avg_monthly_purchase = round(data['Monthly Purchase (Thou. IDR)'].mean(), 2)
        st.markdown(f"<h2 style='color: lightgreen; font-size: 36px; text-align: left;'>IDR {avg_monthly_purchase} K</p>", unsafe_allow_html=True)
        
    # Create a row for layout
    row2 = st.columns(2)

    # Row 2, Column 1
    with row2[0]:

        # Customers per Product
        st.subheader("Customers per Product")
        categories = ['Games Product', 'Music Product', 'Education Product', 'Video Product', 'Use MyApp']
        results = {}
        for column in categories:
            results[column] = data[column].eq('Yes').sum()
        result_series = pd.Series(results)

        # Create a bar chart using Plotly
        fig = px.bar(result_series, x=result_series.index, y=result_series.values, labels={'x': 'Product', 'y': 'Total Customers'})
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

    # Row 2, Column 2
    with row2[1]:
        # Grouping Monthly Purchase
        st.subheader("Grouping Monthly Purchase")
        categories = ['High Purchase', 'Mid Purchase', 'Low Purchase']
        value = {'High Purchase': 0, 'Mid Purchase': 0, 'Low Purchase': 0}
        for i, row in data.iterrows():
            if row['Monthly Purchase (Thou. IDR)'] <= 50:
                value['Low Purchase'] += 1
            elif 50 < row['Monthly Purchase (Thou. IDR)'] <= 100:
                value['Mid Purchase'] += 1
            elif row['Monthly Purchase (Thou. IDR)'] > 100:
                value['High Purchase'] += 1
        value = pd.Series(value)

        # Create a bar chart using Plotly
        fig = px.bar(value, x=categories, y=value.values, labels={'x': 'Monthly Purchase Categories', 'y': 'Total Customers'})
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

    # Create a row for layout
    row3 = st.columns(2)

    # Row 3, Column 1
    with row3[0]:
        # Customers per Product dan Location Distribution
        # Create a row for layout
        row4 = st.columns(1)

        # Row 4, Column 1
        with row4[0]:
            # Location Distribution
            st.subheader("Location Distribution")
            location_distribution = data['Location'].value_counts()
            fig = px.pie(names=location_distribution.index, values=location_distribution, title="Location Distribution")
            st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

            # Row 4, Column 2 (moved from row5[0] to row4[0])
            # Device Class Distribution
            st.subheader("Device Class Distribution")
            device_class_counts = data['Device Class'].value_counts()
            fig = px.bar(x=device_class_counts.index, y=device_class_counts.values, labels={'x': 'Device Class', 'y': 'Total Customers'})
            st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

    # Row 3, Column 2
    with row3[1]:
        # Distribution Based on Payment Method
        st.subheader("Distribution Based on Payment Method")
        payment_distribution = data['Payment Method'].value_counts()
        fig = px.pie(names=payment_distribution.index, values=payment_distribution, title="Payment Method User Percentage")
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar



elif selected_option == "Overview Churn":
    import plotly.express as px
    
    # Filter churned and not churned data
    churn_data = data[data['Churn Label'] == 'Yes']
    not_churn_data = data[data['Churn Label'] == 'No']

    col1, col2 = st.columns(2)

    with col1:

        # Calculate churn rates by Education Product
        call_center_cat = ["No internet service", 'No', 'Yes']
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Education Product'] == call_center_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Education Product'] == call_center_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Education Product'] == call_center_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Education Product'] == call_center_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Education Product'] == call_center_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Education Product'] == call_center_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df = pd.DataFrame({'x': x * 3, 'Education Product': call_center_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart using Plotly
        fig = px.bar(data_df, x='x', y='Churn Rate', color='Education Product', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Education Product")

        # Customize the plot
        fig.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Calculate churn rates by Use MyApp
        use_myapp_cat = ["No", "Yes", "No internet service"]  # Tambahkan "No internet service" ke kategori
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Use MyApp'] == use_myapp_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Use MyApp'] == use_myapp_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Use MyApp'] == use_myapp_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Use MyApp'] == use_myapp_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Use MyApp'] == use_myapp_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Use MyApp'] == use_myapp_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_use_myapp = pd.DataFrame({'x': x * 3, 'Use MyApp': use_myapp_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Use MyApp using Plotly
        fig_use_myapp = px.bar(data_df_use_myapp, x='x', y='Churn Rate', color='Use MyApp', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Use MyApp")

        # Customize the plot
        fig_use_myapp.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_use_myapp, use_container_width=True)


        # Calculate churn rates by Video Product
        video_product_cat = ["No", "Yes", "No internet service"]  # Tambahkan "No internet service" ke kategori
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Video Product'] == video_product_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Video Product'] == video_product_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Video Product'] == video_product_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Video Product'] == video_product_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Video Product'] == video_product_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Video Product'] == video_product_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_video_product = pd.DataFrame({'x': x * 3, 'Video Product': video_product_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Video Product using Plotly
        fig_video_product = px.bar(data_df_video_product, x='x', y='Churn Rate', color='Video Product', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Video Product")

        # Customize the plot
        fig_video_product.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_video_product, use_container_width=True)


        # Calculate churn rates by Music Product
        music_product_cat = ["No", "Yes", "No internet service"]  # Tambahkan "No internet service" ke kategori
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Music Product'] == music_product_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Music Product'] == music_product_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Music Product'] == music_product_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Music Product'] == music_product_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Music Product'] == music_product_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Music Product'] == music_product_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_music_product = pd.DataFrame({'x': x * 3, 'Music Product': music_product_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Music Product using Plotly
        fig_music_product = px.bar(data_df_music_product, x='x', y='Churn Rate', color='Music Product', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Music Product")

        # Customize the plot
        fig_music_product.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_music_product, use_container_width=True)



    with col2:

        # Calculate churn rates by Games Product
        games_product_cat = ["No", "Yes", "No internet service"]  # Tambahkan "No internet service" ke kategori
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Games Product'] == games_product_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Games Product'] == games_product_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Games Product'] == games_product_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Games Product'] == games_product_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Games Product'] == games_product_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Games Product'] == games_product_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_games_product = pd.DataFrame({'x': x * 3, 'Games Product': games_product_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Games Product using Plotly
        fig_games_product = px.bar(data_df_games_product, x='x', y='Churn Rate', color='Games Product', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Games Product")

        # Customize the plot
        fig_games_product.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_games_product, use_container_width=True)


        # Calculate churn rates by Call Center
        call_center_cat = ["No internet service", 'No', 'Yes']
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Call Center'] == call_center_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Call Center'] == call_center_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Call Center'] == call_center_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Call Center'] == call_center_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Call Center'] == call_center_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Call Center'] == call_center_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_call_center = pd.DataFrame({'x': x * 3, 'Call Center': call_center_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Call Center using Plotly
        fig_call_center = px.bar(data_df_call_center, x='x', y='Churn Rate', color='Call Center', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Call Center")

        # Customize the plot
        fig_call_center.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_call_center, use_container_width=True)

        # Kategorikan Customer Tenure Class
        tenure_class_cat = ['New Customer', 'Regular Customer', 'Long-Term Customer']
        x = ['Active', 'Churned']

        # Hitung Churn Rate untuk masing-masing Customer Tenure Class
        y1 = [
            len(not_churn_data[not_churn_data['Customer Tenure Class'] == tenure_class_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Customer Tenure Class'] == tenure_class_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Customer Tenure Class'] == tenure_class_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Customer Tenure Class'] == tenure_class_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Customer Tenure Class'] == tenure_class_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Customer Tenure Class'] == tenure_class_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_tenure_class = pd.DataFrame({'x': x * 3, 'Customer Tenure Class': tenure_class_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Customer Tenure Class using Plotly
        fig_tenure_class = px.bar(data_df_tenure_class, x='x', y='Churn Rate', color='Customer Tenure Class', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Customer Tenure Class")

        # Customize the plot
        fig_tenure_class.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_tenure_class, use_container_width=True)

        # Calculate churn rates by Device Class
        device_class_cat = ["Mid End", "High End", "Low End"]
        x = ['Active', 'Churned']

        y1 = [
            len(not_churn_data[not_churn_data['Device Class'] == device_class_cat[0]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Device Class'] == device_class_cat[0]]) / len(churn_data) * 100
        ]
        y2 = [
            len(not_churn_data[not_churn_data['Device Class'] == device_class_cat[1]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Device Class'] == device_class_cat[1]]) / len(churn_data) * 100
        ]
        y3 = [
            len(not_churn_data[not_churn_data['Device Class'] == device_class_cat[2]]) / len(not_churn_data) * 100,
            len(churn_data[churn_data['Device Class'] == device_class_cat[2]]) / len(churn_data) * 100
        ]

        # Create a DataFrame for the data
        data_df_device_class = pd.DataFrame({'x': x * 3, 'Device Class': device_class_cat * 2, 'Churn Rate': y1 + y2 + y3})

        # Create a stacked bar chart for Churn Rate by Device Class using Plotly
        fig_device_class = px.bar(data_df_device_class, x='x', y='Churn Rate', color='Device Class', barmode='relative',
                    labels={'x': 'Customer Status', 'Churn Rate': 'Churn Rate (Percentage)'},
                    title="Churn Rate by Device Class")

        # Customize the plot
        fig_device_class.update_traces(marker_line_width=0)

        # Display the Plotly chart using Streamlit
        st.plotly_chart(fig_device_class, use_container_width=True)







elif selected_option == "Filter by Location":
    # Filter by location
    locations = data['Location'].unique()
    selected_location = st.sidebar.selectbox("Select a location", locations)
    filtered_data = data[data['Location'] == selected_location]

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Column 1
    with col1:
        # Filtered Data for Location
        st.subheader("Filtered Data for Location:")
        st.markdown(f"<h2 style='color: lightgreen;'>{selected_location}</h2>", unsafe_allow_html=True)

        # Number of Customers for the selected location
        st.subheader("Number of Customers")
        st.markdown(f"<h1 style='color: lightgreen; font-size: 48px; text-align: left;'>{filtered_data['Customer ID'].count()}</h1>", unsafe_allow_html=True)

        # Number of Churned Customers for the selected location
        st.subheader("Number of Churned Customers")
        churn_count = filtered_data['Churn Label'].value_counts().get('Yes', 0)
        st.markdown(f"<h1 style='color: lightgreen; font-size: 48px; text-align: left;'>{churn_count}</h1>", unsafe_allow_html=True)

        # Churn Rate for the selected location
        st.subheader("Churn Rate")
        churn_rate = round(churn_count / filtered_data['Customer ID'].count() * 100, 2)
        st.markdown(f"<p style='color: lightgreen; font-size: 48px; text-align: left;'>{churn_rate}%</p>", unsafe_allow_html=True)

        # Average Monthly Purchase for the selected location
        st.subheader("Average Purchase")
        avg_monthly_purchase = round(filtered_data['Monthly Purchase (Thou. IDR)'].mean(), 2)
        st.markdown(f"<p style='color: lightgreen; font-size: 24px; text-align: left;'>IDR {avg_monthly_purchase} K</p>", unsafe_allow_html=True)

        # Customers per Product for the selected location
        st.subheader("Customers per Product")
        categories = ['Games Product', 'Music Product', 'Education Product', 'Video Product', 'Use MyApp']
        results = {}
        for column in categories:
            results[column] = filtered_data[column].eq('Yes').sum()
        result_series = pd.Series(results)

        # Create a bar chart using Plotly
        fig = px.bar(result_series, x=result_series.index, y=result_series.values, labels={'x': 'Product', 'y': 'Total Customers'})
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

    # Column 2
    with col2:
        # Distribution Based on Payment Method for the selected location
        st.subheader("Distribution Based on Payment Method")
        payment_distribution = filtered_data['Payment Method'].value_counts()
        fig = px.pie(names=payment_distribution.index, values=payment_distribution, title="Payment Method User Percentage")
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar

        # Device Class Distribution for the selected location
        st.subheader("Device Class Distribution")
        device_class_counts = filtered_data['Device Class'].value_counts()
        fig = px.bar(x=device_class_counts.index, y=device_class_counts.values, labels={'x': 'Device Class', 'y': 'Total Customers'})
        st.plotly_chart(fig, use_container_width=True)  # Lebar maksimal setengah dari lebar layar


# Load the churn prediction model
model_path = 'churn_pred_model.pkl'  # Ganti dengan path yang benar ke model yang Anda simpan
with open(model_path, 'rb') as file:
    model = pickle.load(file)

if selected_option == "Prediction":
    # Customer Prediction
    st.header("Customer Churn Prediction")
    st.write("Please provide customer information:")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    # Left column
    with col1:
        tenure_months = st.number_input("Enter Tenure Months", min_value=0, value=0)
        # Assuming you have preloaded 'data' variable containing unique values for the dropdowns
        location = st.selectbox("Select Location", ['Jakarta'])  # Since all values are 'Jakarta'
        device_class = st.selectbox("Select Device Class", ['Mid End', 'High End', 'Low End'])
        games_product = st.radio("Select Games Product", ["Yes", "No"])
        music_product = st.radio("Select Music Product", ["Yes", "No"])

    # Right column
    with col2:
        education_product = st.radio("Select Education Product", ["Yes", "No"])
        video_product = st.radio("Select Video Product", ["Yes", "No"])
        use_myapp = st.radio("Select Use MyApp", ["Yes", "No"])
        payment_method = st.selectbox("Select Payment Method", ['Pulsa', 'Digital Wallet', 'Debit', 'Credit'])
        monthly_purchase = st.number_input("Enter Monthly Purchase (Thou. IDR)", min_value=0.0, value=0.0)

    # Button to calculate churn prediction
    if st.button("Predict Churn"):
        # Prepare the data for prediction
        input_data = pd.DataFrame([[tenure_months, location, device_class, games_product, music_product, 
                                    education_product, video_product, use_myapp, payment_method, monthly_purchase]],
                                  columns=['Tenure Months', 'Location', 'Device Class', 'Games Product', 'Music Product',
                                           'Education Product', 'Video Product', 'Use MyApp', 'Payment Method', 'Monthly Purchase'])
        
        # Perform churn prediction
        churn_probability = model.predict_proba(input_data)[0][1]

        st.subheader("The predicted probability of churn is:")

        if churn_probability < 0.5:
            st.markdown(f"<h1 style='font-size: 48px; font-weight: bold; color: lightgreen;'>{churn_probability * 100:.2f}% chance of not churning</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='font-size: 48px; font-weight: bold; color: red;'>{churn_probability * 100:.2f}% chance of churning</h1>", unsafe_allow_html=True)