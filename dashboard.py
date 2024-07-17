import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
# day_df = pd.read_csv("day.csv")
# hour_df = pd.read_csv("hour.csv")
@st.cache_data
def load_data():
    if not os.path.exists("day.csv") or not os.path.exists("hour.csv"):
        st.error("Files not found. Please make sure 'day.csv' and 'hour.csv' are in the current directory.")
        return None, None
    day_df = pd.read_csv("day.csv")
    hour_df = pd.read_csv("hour.csv")
    return day_df, hour_df

day_df, hour_df = load_data()
if day_df is None or hour_df is None:
    st.stop()

# Set the style for the plots
sns.set(style="whitegrid")

# Sidebar for user input
st.sidebar.title("Bike Sharing Dashboard")
st.sidebar.markdown("### Select filters and options")

# Year selection
year = st.sidebar.selectbox("Select Year", [2011, 2012])

# Color theme selection
color_theme = st.sidebar.selectbox("Select Color Theme", ['pastel', 'muted', 'dark'])

# Filter data based on the selected year
day_df = day_df[day_df['yr'] == (year - 2011)]
hour_df = hour_df[hour_df['yr'] == (year - 2011)]

# Define a function to create bar plots for average rentals
def bar_plot_avg(df, group_by, title, xlabel, ylabel, xticklabels):
    avg_data = df.groupby(group_by)['cnt'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_data.plot(kind='bar', color=sns.color_palette(color_theme), ax=ax)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=20)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=0)
    st.pyplot(fig)

# Define a function to create line plots for monthly rental trends
def dual_line_plot_monthly(df, title, ylabel):
    monthly_data_casual = df.groupby('mnth')['casual'].mean()
    monthly_data_registered = df.groupby('mnth')['registered'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_data_casual.plot(kind='line', marker='o', ax=ax, label='Casual', color='blue')
    monthly_data_registered.plot(kind='line', marker='o', ax=ax, label='Registered', color='green')
    
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Month', fontweight='bold', labelpad=20)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    st.pyplot(fig)

# Define a function to create the average hourly bike rentals plot
def plot_average_hourly_rentals(df, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x='hr', y='cnt', data=df, estimator='mean', ci=None, ax=ax)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=20)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20)
    st.pyplot(fig)

# Function to add navigation links
def add_nav_links():
    st.sidebar.markdown("[Introduction](#introduction)")
    st.sidebar.markdown("[Seasonal Usage Patterns](#seasonal-usage-patterns)")
    st.sidebar.markdown("[Impact of Weather Conditions](#impact-of-weather-conditions)")
    st.sidebar.markdown("[Weekday vs Weekend Usage](#weekday-vs-weekend-usage)")
    st.sidebar.markdown("[Effect of Holidays](#effect-of-holidays)")
    st.sidebar.markdown("[User Type Analysis](#user-type-analysis)")
    st.sidebar.markdown("[Average Hourly Bike Rentals](#average-hourly-bike-rentals)")
    st.sidebar.markdown("[Correlation Matrix](#correlation-matrix)")
    st.sidebar.markdown("[Business Improvements](#business-improvements)")
    st.sidebar.markdown("[[Machine Learning] Clustering Analysis](#clustering-analysis)")
    st.sidebar.markdown("[[Machine Learning] Event and Anomaly Detection](#event-and-anomaly-detection)")
    st.sidebar.markdown("[[Machine Learning] Prediction for 2013](#prediction)")

# Add navigation links
st.sidebar.markdown("### Select Navigation")
navigation = add_nav_links()

# Title of the dashboard
st.markdown("<a name='introduction'></a>", unsafe_allow_html=True)
st.title('Bike Sharing Data Analysis Dashboard')

# Introduction
st.markdown("""
This interactive dashboard provides insights and visualizations based on the bike sharing data analysis. 
Explore the patterns and trends in bike rentals and understand the factors influencing them.

### Background:
Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, environmental and health issues. 

### Analysis Goals:
- **Seasonal Usage Patterns**: How does bike rental usage vary across different seasons (spring, summer, fall, winter) over the two years (2011 and 2012)?
- **Impact of Weather Conditions**: What is the impact of different weather conditions (clear, mist, light snow/rain, heavy rain/snow) on the number of bike rentals?
- **Weekday vs Weekend Usage**: How does bike rental usage differ between weekdays and weekends?
- **Effect of Holidays on Bike Rentals**: How do holidays affect the number of bike rentals compared to non-holiday weekdays?
- **User Type Analysis (Casual vs Registered)**: What are the differences in usage patterns between casual users and registered users over different months?

### Criteria for Effective Analysis:
- **Specific**: Each analysis focuses on a specific aspect of the bike-sharing data.
- **Measurable**: The analyses use measurable data points, such as rental counts.
- **Action-oriented**: The insights can guide business decisions and operational improvements.
- **Relevant**: The analyses are relevant to understanding user behavior and optimizing operations.
- **Time-bound**: The data covers the years 2011 and 2012.

### Dataset:
Before conducting the analysis, the dataset has been cleaned, such as checking for incompatible data types, checking for null values, duplicate rows, incompatible values, and so on.
""")
st.dataframe(day_df.head())
st.dataframe(hour_df.head())

# Seasonal Usage Patterns
st.markdown("<a name='seasonal-usage-patterns'></a>", unsafe_allow_html=True)
st.header('Seasonal Usage Patterns')
st.markdown("""
**Analysis Goal**: To understand how bike rentals vary across different seasons.
""")
with st.spinner('Processing...'):
    bar_plot_avg(day_df, 'season', 'Average Bike Rentals per Season', 'Season', 'Average Rentals', ['Spring', 'Summer', 'Fall', 'Winter'])
st.markdown("""
**Analysis Result**: The highest bike rentals occur in the fall, followed by summer. Winter has the lowest number of rentals, indicating that warmer weather encourages more people to rent bikes.
""")

# Impact of Weather Conditions
st.markdown("<a name='impact-of-weather-conditions'></a>", unsafe_allow_html=True)
st.header('Impact of Weather Conditions')
st.markdown("""
**Analysis Goal**: To analyze the impact of different weather conditions on bike rentals.
""")
with st.spinner('Processing...'):
    bar_plot_avg(day_df, 'weathersit', 'Average Bike Rentals by Weather Condition', 'Weather Condition', 'Average Rentals', ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain/Snow'])
st.markdown("""
**Analysis Result**: Clear weather conditions result in the highest number of bike rentals, while heavy rain or snow significantly reduces the number of rentals.
""")

# Weekday vs Weekend Usage
st.markdown("<a name='weekday-vs-weekend-usage'></a>", unsafe_allow_html=True)
st.header('Weekday vs Weekend Usage')
st.markdown("""
**Analysis Goal**: To compare bike rental usage between weekdays and weekends.
""")
day_df['is_weekend'] = day_df['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)
with st.spinner('Processing...'):
    bar_plot_avg(day_df, 'is_weekend', 'Average Bike Rentals: Weekday vs Weekend', 'Day Type', 'Average Rentals', ['Weekday', 'Weekend'])
st.markdown("""
**Analysis Result**: Weekdays show a higher average number of rentals compared to weekends, indicating that people are more likely to rent bikes for daily activity, such as the need for transportation while working.
""")

# Effect of Holidays on Bike Rentals
st.markdown("<a name='effect-of-holidays'></a>", unsafe_allow_html=True)
st.header('Effect of Holidays on Bike Rentals')
st.markdown("""
**Analysis Goal**: To assess how holidays affect bike rentals.
""")
with st.spinner('Processing...'):
    bar_plot_avg(day_df, 'holiday', 'Average Bike Rentals on Holidays vs Non-Holidays', 'Holiday', 'Average Rentals', ['Non-Holiday', 'Holiday'])
st.markdown("""
**Analysis Result**: Holidays tend to have a lower number of bike rentals compared to weekdays, which reinforces the previous statement that more people use bike rentals for daily activities.
""")

# User Type Analysis (Casual vs Registered)
st.markdown("<a name='user-type-analysis'></a>", unsafe_allow_html=True)
st.header('User Type Analysis')
st.markdown("""
**Analysis Goal**: To compare rental trends between casual users and registered users over different months.
""")
with st.spinner('Processing...'):
    dual_line_plot_monthly(day_df, 'Monthly Rental Trends for Casual and Registered Users', 'Average Rentals')
st.markdown("""
**Analysis Result**: Comparing 2011 and 2012, we see a general increase in bike rentals in 2012 for both casual and registered users. This indicates a growing trend in the adoption of the bike-sharing system. The patterns remain similar, with peaks in summer for casual users and consistent usage by registered users throughout the year.
""")

# Average Hourly Bike Rentals
st.markdown("<a name='average-hourly-bike-rentals'></a>", unsafe_allow_html=True)
st.header('Average Hourly Bike Rentals')
st.markdown("""
**Analysis Goal**: To identify the hourly patterns in bike rentals throughout the day.
""")
with st.spinner('Processing...'):
    plot_average_hourly_rentals(hour_df, 'Average Hourly Bike Rentals', 'Hour of the Day', 'Average Rentals')
st.markdown("""
**Analysis Result**: The data shows that the peak rental times are in the morning and evening, which is the peak time for people going to and from work or school.
""")

# Correlation Matrix
st.markdown("<a name='correlation-matrix'></a>", unsafe_allow_html=True)
st.header('Correlation Matrix')
st.markdown("""
**Analysis Goal**: To explore the relationships between different features in the dataset.
""")
with st.spinner('Processing...'):
    day_df_copy = day_df.copy()
    day_df_copy.drop('yr', axis=1, inplace=True)
    numeric_df = day_df_copy.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(14, 10))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, annot_kws={"size": 10}, fmt=".2f")
    ax.set_title('Correlation Matrix of Day Dataset Features', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    st.pyplot(fig)
st.markdown("""
**Analysis Result**: The correlation matrix reveals strong positive correlations between registered users and total rentals, and between temperature and total rentals. Warmer weather likely encourages more people to rent bikes, as biking is more pleasant and feasible in such conditions. There is a negative correlation between weather conditions and bike rentals, indicating that worse weather conditions reduce bike rentals.
""")

# Conclusion and Business Improvements
st.markdown("<a name='business-improvements'></a>", unsafe_allow_html=True)
st.header('Business Improvements')
st.markdown("""
Based on the analysis above, the following business improvements and recommendations can be made:
- Increase bike availability and perform proactive maintenance before and during the summer and fall seasons to accommodate higher demand. Implement marketing campaigns targeting these peak seasons to maximize rentals.
- Develop a real-time weather-responsive pricing strategy, offering discounts or incentives on days with adverse weather conditions to encourage usage. Ensure bikes are weatherproof and provide weather updates to users through the app.
- Optimize bike distribution to ensure more bikes are available in commercial and residential areas during weekdays. Implement targeted promotions for weekend leisure activities to balance usage.
- Create special holiday promotions or events to attract users. Partner with local businesses and tourist attractions to offer combined deals that include bike rentals.
- Enhance the registration process and offer incentives for casual users to become registered users. Provide loyalty programs and exclusive benefits to registered users to retain them.
- Ensure higher availability of bikes during peak hours and strategically place bikes in high-demand areas. Consider implementing a dynamic pricing model to manage demand during peak times.
- Use the insights from the correlation matrix to predict demand based on weather forecasts and user registration data. Implement data-driven decision-making to optimize bike availability and maintenance schedules.  
            
Implementing these improvements based on the analysis will help enhance the overall user experience, increase bike rental usage, and optimize operational efficiency. By understanding the patterns and factors influencing bike rentals, the bike-sharing program can better cater to user needs and adapt to changing conditions.
However, further analysis using other datasets, such as internal company data, is needed.
""")

# Clustering Analysis
st.markdown("<a name='clustering-analysis'></a>", unsafe_allow_html=True)
st.header('Clustering Analysis')
st.markdown("""
**Analysis Goal**: To identify distinct patterns in bike rentals using clustering using K-Means.
""")

with st.spinner('Processing...'):
    # Prepare data for clustering
    clustering_df = day_df[['temp', 'cnt']]
    scaler = StandardScaler()
    scaled_clustering_df = scaler.fit_transform(clustering_df)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_clustering_df)

    # Get the centroids and sort them by the temperature feature (first feature)
    centroids = kmeans.cluster_centers_
    sorted_centroids_indices = np.argsort(centroids[:, 0])

    # Create a mapping from original cluster labels to new labels
    label_mapping = {sorted_centroids_indices[i]: i for i in range(len(sorted_centroids_indices))}

    # Map the original cluster labels to the new labels
    new_clusters = np.vectorize(label_mapping.get)(clusters)
    clustering_df['Cluster'] = new_clusters

    # Define the color palette based on the selected color theme
    if color_theme == 'pastel':
        palette = sns.color_palette("pastel", n_colors=3)
    elif color_theme == 'muted':
        palette = sns.color_palette("muted", n_colors=3)
    elif color_theme == 'dark':
        palette = sns.color_palette("dark", n_colors=3)
    else:
        palette = sns.color_palette("deep", n_colors=3)  # Default palette

    # Visualize the clusters with custom colors
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.scatterplot(x='temp', y='cnt', hue='Cluster', palette=palette, data=clustering_df, ax=ax)
    ax.set_title('Clusters of Bike Rentals based on Temperature and Count', fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Normalized Temperature', fontweight='bold', labelpad=20)
    ax.set_ylabel('Total Bike Rentals', fontweight='bold', labelpad=20)
    st.pyplot(fig)

st.markdown("""
**Analysis Result**: The clustering analysis identifies three distinct clusters based on the features. This segmentation helps in understanding different patterns in bike rentals.
- **Cluster 0**: Represents low rentals, often associated with lower temperatures and adverse weather conditions.
- **Cluster 1**: Represents moderate rentals, possibly indicating average weather conditions.
- **Cluster 2**: Represents high rentals, associated with higher temperatures and favorable weather conditions.

**Implications for Business**:
- **Targeted Promotions**: During colder temperatures (low rentals), promotions or discounts could be offered to incentivize bike rentals. This could help in balancing out the demand and increasing rentals during low-temperature periods. During warmer temperatures (high rentals), the bike-sharing service can ensure higher availability of bikes, as the demand is significantly higher. Promotional campaigns can be focused on maintenance and ensuring bike availability rather than discounts.
- **Resource Allocation**: More resources can be allocated to areas with lower temperatures to promote bike usage. For example, adding features like bike heaters or providing better infrastructure in colder areas can help increase rentals. Ensure that the bikes are well-maintained and readily available in areas experiencing higher temperatures. Strategic placement and resource allocation can help meet the high demand efficiently.
- **Marketing Strategies**: Marketing strategies to encourage biking in colder weather. This could include educational campaigns about the benefits of biking in various weather conditions, tips for staying warm, and highlighting bike-friendly routes. Emphasize the convenience and benefits of biking during warm weather. Highlight special routes, scenic tours, and health benefits to attract more users.
- **Operational Efficiency**: By understanding the patterns of bike rentals across different temperatures, the bike-sharing service can optimize their operations, maintenance schedules, and resource distribution. For example, more frequent bike checks and maintenance might be necessary during high rental periods (warmer temperatures).
""")

# Event and Anomaly Detection using TensorFlow
st.markdown("<a name='event-and-anomaly-detection'></a>", unsafe_allow_html=True)
st.header('Event and Anomaly Detection using TensorFlow')
st.markdown("""
**Analysis Goal**: To detect events and anomalies in bike rentals data using time-series analysis with TensorFlow.
""")

with st.spinner('Processing...'):
    # Prepare data for LSTM model
    df = day_df[['dteday', 'cnt']]
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.set_index('dteday', inplace=True)
    df = df.resample('D').sum()

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 30
    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=5)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Inverse transform the predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Plot the results
    train_data = df[:split + seq_length]
    test_data = df[split + seq_length:]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(train_data.index, train_data['cnt'], label='Training Data')
    ax.plot(test_data.index, test_data['cnt'], label='Test Data')
    ax.plot(test_data.index, test_predictions, label='Predictions', color='red')
    ax.set_title('Bike Rentals Predictions using LSTM', fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Date', fontweight='bold', labelpad=20)
    ax.set_ylabel('Total Bike Rentals', fontweight='bold', labelpad=20)
    ax.legend()
    st.pyplot(fig)

    # Highlight significant events and anomalies
    st.markdown("""
    **Analysis Result**: The LSTM model can be used to detect significant events and anomalies in the bike rentals data. For example:
    - **2012-10-29**: The data shows a significant drop in bike rentals due to Hurricane Sandy.
    - **Other Events**: The model can help identify other significant events affecting bike rentals by examining the anomalies in the data.

    **Implications for Business**:
    - **Event Planning**: We can use the model to plan for significant events and adjust bike availability accordingly.
    - **Anomaly Detection**: Identify unusual patterns and take corrective actions to address potential issues.
    """)

    day_df['cnt_scaled'] = scaler.fit_transform(day_df[['cnt']])

    # Splitting data into training and test sets
    train_size = int(len(day_df) * 0.8)
    train_data = day_df['cnt_scaled'][:train_size].values.reshape(-1, 1)
    test_data = day_df['cnt_scaled'][train_size:].values.reshape(-1, 1)

    # Build Autoencoder Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation=None)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the Model
    history = model.fit(train_data, train_data, epochs=50, batch_size=16, validation_split=0.1, shuffle=True)

    # Detect Anomalies
    train_pred = model.predict(train_data)
    train_mse = np.mean(np.power(train_data - train_pred, 2), axis=1)

    test_pred = model.predict(test_data)
    test_mse = np.mean(np.power(test_data - test_pred, 2), axis=1)

    threshold = np.max(train_mse)
    day_df['anomaly_tf'] = np.append(train_mse, test_mse) > threshold
    day_df['reconstruction_error'] = np.append(train_mse, test_mse)

    # Separate data into anomalies and normal days
    anomalies = day_df[day_df['anomaly_tf']]
    normal_days = day_df[~day_df['anomaly_tf']]

    # Plot the anomalies
    fig, ax = plt.subplots()
    ax.plot(day_df['dteday'], day_df['cnt'], label='Total Rent')
    ax.plot(anomalies['dteday'], anomalies['cnt'], 'ro', label='Anomalies')
    ax.legend()
    st.pyplot(fig)

    # Display details of anomalies
    st.write("Detailed Data on Anomaly Dates:")
    st.write(anomalies)

    if year == 2011:
        # Highlight significant events and anomalies
        st.markdown("""
        **Analysis Result**: There is no anomalies.
        """)

    if year == 2012:
        # Highlight significant events and anomalies
        st.markdown("""
        **Analysis Result**:
        On October 29, 2012, the number of bike rentals dropped significantly to 22 due to bad weather conditions such as light snow and thunderstorms, as people tend to avoid biking in such conditions. On December 24, 2012, rentals were lower at 920 as people were likely busy with Christmas preparations, leading to less time for biking. The trend continued on December 25, 2012, with rentals at 1013, suggesting that despite it being Christmas, people preferred other activities or modes of transport. On December 26, 2012, the day after Christmas, rentals dropped to 441, likely due to a return to normal activities coupled with bad weather conditions such as light snow and thunderstorms.

        **Implications for Business**:
        - **Operational Planning**: Improve service during holiday periods by possibly offering special promotions.
        - **Weather Information**: Provide real-time weather information to users to help them plan their trips better and consider offering incentives on bad weather days to maintain rental numbers.
        - **Further Analysis**: Conduct additional analysis with historical weather data, city events, and policy changes to gain deeper insights into factors affecting bike usage.
        - **Adaptive Strategies**: Develop adaptive strategies to seasonal changes and weather conditions to ensure optimal bike rental service throughout the year.

        """)

# Prediction using Neural Network
st.markdown("<a name='prediction'></a>", unsafe_allow_html=True)
st.header('Prediction using scikit-learn')
st.markdown("""
**Analysis Goal**:
The goal of this analysis is to predict the total number of bike rentals for the second half of 2012 using MLP Regressor. The model is trained on historical data from 2011 and the first half of 2012 to understand the patterns and trends in bike rentals and then used to make predictions for the second half of 2012.
""")

with st.spinner('Processing...'):
    day_data = pd.read_csv('day.csv')

    day_data['dteday'] = pd.to_datetime(day_data['dteday'])

    # Filter data for training (2011 and first half of 2012) and testing (second half of 2012)
    train_data = day_data[(day_data['yr'] == 0) | ((day_data['yr'] == 1) & (day_data['mnth'] <= 6))]
    test_data = day_data[(day_data['yr'] == 1) & (day_data['mnth'] > 6) & (day_data['dteday'] <= '2012-12-31')]

    # Define features and target
    features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Increase model complexity
    model = MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        max_iter=2000,
        random_state=42,
        learning_rate='adaptive',  # 'constant' or 'adaptive'
        learning_rate_init=0.0001,  # Lower learning rate
        solver='adam'  # Optimizer can be 'lbfgs', 'sgd', or 'adam'
    )

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Make predictions for the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Add predictions to the test data
    test_data['predicted_cnt'] = y_pred

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(test_data['dteday'], test_data['cnt'], label='Actual')
    plt.plot(test_data['dteday'], test_data['predicted_cnt'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Total Rentals')
    plt.title('Actual vs Predicted Bike Rentals')
    plt.legend()
    st.pyplot(plt)
st.markdown("""
    **Analysis Result**:
    - The model successfully captures the general pattern of bike rentals. Both the actual and predicted values follow similar trends over time, indicating that the model has learned the underlying patterns in the data.
    - The model seems to capture seasonal variations effectively. For instance, bike rentals typically increase during warmer months and decrease during colder months. This seasonal pattern is reflected in the predictions.
    """)

# # Load the datasets
# df_day_pred = pd.read_csv('day.csv')

# # Normalize the features and target variable
# features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
# scaler_features = MinMaxScaler()
# df_day_pred[features] = scaler_features.fit_transform(df_day_pred[features])
# df_day_pred['cnt_scaled'] = scaler.fit_transform(df_day_pred[['cnt']])

# # Split data into training and test sets
# train_df = df_day_pred[(df_day_pred['yr'] == 0) | ((df_day_pred['yr'] == 1) & (df_day_pred['mnth'] <= 6))]  # 2011 and first half of 2012
# test_df = df_day_pred[(df_day_pred['yr'] == 1) & (df_day_pred['mnth'] > 6)]  # second half of 2012

# train_x = train_df[features].values
# train_y = train_df['cnt_scaled'].values
# test_x = test_df[features].values
# test_y = test_df['cnt_scaled'].values

# # Create tf.data.Dataset objects
# train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).batch(32)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

# # Build Neural Network Model
# model = Sequential([
#     tf.keras.layers.Input(shape=(train_x.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     Dropout(0.2),
#     tf.keras.layers.Dense(64, activation='relu'),
#     Dropout(0.2),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # Add callbacks for early stopping and saving the best model
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=True)
# ]

# # Train the Model
# history = model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=callbacks)

# # Evaluate the Model
# loss, mae = model.evaluate(test_dataset)
# st.write(f'Mean Absolute Error on Test Set: {mae}')

# # Predict for the test set
# predictions_scaled = model.predict(test_x)
# predictions = scaler.inverse_transform(predictions_scaled)

# # Add predictions to the dataframe
# test_df['predicted_cnt'] = predictions

# # Visualization
# plt.figure(figsize=(14, 7))
# plt.plot(test_df['dteday'], test_df['cnt'], label='Actual Count')
# plt.plot(test_df['dteday'], test_df['predicted_cnt'], label='Predicted Count', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('Total Rentals')
# plt.title('Bike Rentals Prediction for the Second Half of 2012')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Save the figure and display in Streamlit
# plt.savefig('prediction_plot.png')
# st.image('prediction_plot.png')

# # Streamlit Visualization
# st.line_chart(test_df.set_index('dteday')[['cnt', 'predicted_cnt']])
