import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import logging

# Loop over each numeric column to plot histograms
def num_plot(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=10, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        
# Loop over each categorical column
def category_plot(df):
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[column].nunique()
        
        if unique_count > 1:  # More than 1 unique value
            value_counts = df[column].value_counts()
            
            if unique_count > 10:  # More than 10 unique values
                # Get the top 10 categories
                value_counts = value_counts.nlargest(25)

            # Plotting
            plt.figure(figsize=(8, 4))
            value_counts.plot(kind='bar', color='skyblue')
            plt.title(f'Bar Chart for {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.75)
            plt.show()
        else:
            print(f"Column '{column}' has only 1 unique value {df[column].unique()}")


def time_series(df,feature):
    # Plotting the time series data
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.plot(df.index, df[feature],  linestyle='-')  
    plt.title('Time Series Data')  # Title of the plot
    plt.xlabel('Date')  # X-axis label
    plt.ylabel(f'{feature}')  # Y-axis label
    plt.grid(True)  # Add grid
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display the plot