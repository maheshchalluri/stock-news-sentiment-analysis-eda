# 📰 Stock News Sentiment Analysis – Part 1: Exploratory Data Analysis & Baseline Model

This repository contains the **first phase** of the *Stock News Sentiment Analysis* project, which explores the relationship between financial news sentiment and stock market behavior.  
The goal of this part is to perform **Exploratory Data Analysis (EDA)**, uncover key insights about the data, and develop a **baseline sentiment prediction model**.

---

## 📘 Project Overview

Financial markets are heavily influenced not only by quantitative indicators like stock prices and trading volume but also by **qualitative factors** such as **investor sentiment**.  
This project aims to bridge the two by analyzing how textual sentiment extracted from financial news correlates with stock movements — using Apple Inc. (AAPL) as the initial focus.

Part 1 lays the analytical foundation through:
- **EDA** of financial and textual features  
- **Data visualization** and correlation analysis  
- **Baseline sentiment model** using Naive Bayes with TF-IDF features  

Subsequent parts of the project will expand into **Word2Vec** and **SentenceTransformer** models for deeper contextual understanding.

---

## 📂 Repository Structure
```
Stock-News-Sentiment-Analysis/
│
├── 📓 Stock news sentiment analysis - EDA - Part 1.ipynb # Main notebook (EDA & baseline model)
│
├── 📁 data/
│ ├── stock_news.csv # Combined news & market data
│
├── 📜 README.md # Project documentation

```

## 🧠 Data Description

The dataset combines **financial news headlines** with corresponding **daily stock price data** for Apple Inc. (AAPL).  
It was constructed from **publicly available sources** including *Kaggle* (financial news sentiment data) and *Yahoo Finance* (historical market data).

| **Column Name** | **Description** |
|------------------|-----------------|
| `Date` | Trading day of the news and stock data |
| `News` | Financial news headline or summary text |
| `Open` | Stock opening price |
| `High` | Highest price of the day |
| `Low` | Lowest price of the day |
| `Close` | Closing price of the day |
| `Volume` | Number of shares traded |
| `Label` | Sentiment polarity — 1 = Positive, 0 = Neutral, -1 = Negative |

---

## 🔍 Exploratory Data Analysis (EDA)

EDA focused on understanding both textual and numerical dimensions of the dataset.

Key observations:
- **Price variables (Open, High, Low, Close)** are *highly correlated*.
- **Trading volume** shows a weak *negative correlation* with prices.
- **News headlines** typically range from *40–60 words*, confirming that the dataset consists of concise summaries.
- **Negative sentiment** is associated with *lower median stock prices*, suggesting potential short-term market impact.

Visualizations included:
- Distribution plots of price and sentiment
- Boxplots of price variables by sentiment label
- Correlation heatmaps for financial metrics
- Time-series plots of price and volume trends

---

## ⚙️ Baseline Model

A **Multinomial Naive Bayes** classifier was implemented as the baseline for sentiment prediction.

### **Pipeline Overview**
1. **Text preprocessing** using TF-IDF vectorization  
2. **Model training** with Multinomial Naive Bayes  
3. **Evaluation** using accuracy, precision, recall, and F1-score  

The model provides an interpretable baseline to compare future NLP-based approaches against.

---

## 🧾 Key Findings

- Stock prices exhibit **strong internal correlation**, reflecting market co-movement.  
- **Volume and price** show a mild inverse trend.  
- **Negative news** generally corresponds with **lower market prices**.  
- The **Naive Bayes baseline model** demonstrates that textual sentiment carries useful predictive information about market direction.

---

## 🚀 Next Steps (Part 2)

The next phase will focus on **enhancing text representation and model performance**:
- Implement **Word2Vec** and **SentenceTransformer** embeddings for richer semantic understanding.
- Compare embedding-based models to the Naive Bayes baseline.
- Explore **weekly sentiment aggregation** to capture lagged market responses.
- Evaluate and visualize performance improvements.

---

## 🧰 Technologies Used

- **Python 3.10+**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, tqdm, yfinance
- **Models:** Multinomial Naive Bayes (Baseline)
- **NLP Techniques:** TF-IDF vectorization, text cleaning, tokenization

---

## 📈 Results Summary

| Metric | Baseline Model (Naive Bayes) |
|---------|------------------------------|
| **Vectorization** | TF-IDF (Unigrams + Bigrams) |
| **Accuracy** | ~Baseline level (depends on dataset split) |
| **Interpretation** | Confirms sentiment-text link |
| **Use Case** | Benchmark for future NLP models |

---

## 🗂️ License

This project is intended for **educational and research purposes only**.  
Data sources are derived from **Kaggle** and **Yahoo Finance** and may be subject to their respective usage terms.

---

## ⭐ Acknowledgements

- [Kaggle Financial News Datasets](https://www.kaggle.com/)  
- [Yahoo Finance API](https://finance.yahoo.com/)  
- [NLTK Sentiment Tools](https://www.nltk.org/)  


---

