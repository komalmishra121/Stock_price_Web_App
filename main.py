import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import streamlit as st


if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("bitcoin.gif")
    st.title("Phase II Project Crystal Ball")
    choice = st.radio("Navigation",
                      ["Upload", "Shape Dimension", "Bitcoin close price", "frequency graph", "boxplot graph",
                       "Pie Chart", "Heat Value"])
    st.info("This interface is built for analysing and comparing models main Dashboard of Project Crystal Ball")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Shape Dimension":
    st.title("Describes Shape Dimension")
    a = df.describe()
    st.dataframe(a)

if choice == "Bitcoin close price":
    st.title("Bitcoin close price")
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.plot(df['Close'])
    ax.set_title('Bitcoin Close Price Graph', fontsize=40)
    ax.set_ylabel('Price in INR')
    st.pyplot(fig)

if choice == "frequency graph":
    st.title("Frequency Graph")
    features = ['Open', 'High', 'Low', 'Adj Close']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, col in enumerate(features):
        sb.distplot(df[col], ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(col)

    st.pyplot(fig)

if choice == "boxplot graph":
    st.title("Boxplot Graph")
    features = ['Open', 'High', 'Low', 'Adj Close']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, col in enumerate(features):
        sb.boxplot(df[col], ax=axes[i // 2, i % 2])
        axes[i // 2, i % 2].set_title(col)
    st.pyplot(fig)

if choice == "Pie Chart":
    df['open-close'] = df['Open'] - df['Adj Close']
    df['low-high'] = df['Low'] - df['High']

    # Create a 'target' column for binary classification labels
    df['target'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1, 0)

    # Display a pie chart to visualize the distribution of 'target' values
    st.title("Target Distribution Pie Chart")
    fig, ax = plt.subplots()
    ax.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

if choice == "Heat Map":
    # Set the figure size
    plt.figure(figsize=(4, 4))

    # Create a heatmap to visualize highly correlated features
    st.title("Highly Correlated Features Heatmap")
    sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)

    # Display the heatmap using Streamlit
    st.pyplot(plt)

