# 📈 Stock Price Prediction using XGBoost and Streamlit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kabilan-1616/Stock-prediction/blob/main/Stock_preditcion.ipynb)

This project demonstrates how to predict stock price movements using technical indicators and machine learning models (XGBoost). It includes a detailed Jupyter Notebook with explanations and a Streamlit app for interactive predictions.

![Image](https://github.com/user-attachments/assets/bcd2fe97-466c-4a19-a801-f459b907982f)


---

## 🔧 Features

- **Jupyter Notebook** (`Stock_prediction.ipynb`):  
  Step-by-step explanation of data loading, feature engineering, model training, and evaluation.

- **Streamlit App** (`stock_prediction.py`):  
  Interactive web app to input stock tickers and get price predictions.

- Technical indicators used:  
  - Simple Moving Average (SMA)  
  - Relative Strength Index (RSI)  
  - Moving Average Convergence Divergence (MACD)

- Models trained:  
  - Next-day closing price prediction (regression)  
  - Next-day price direction (classification)  
  - 3-day price movement direction (classification)

---

## 🚀 How to Run

### 📓 Run the Notebook

You can explore and run the notebook in two ways:

- **Option 1: Run in Google Colab**  
  Click the badge at the top of this README to open the notebook directly in Colab — no installation needed.

- **Option 2: Run Locally**  
  Download and run locally.

  ```bash
  pip install -r requirements.txt
  jupyter notebook Stock_prediction.ipynb
