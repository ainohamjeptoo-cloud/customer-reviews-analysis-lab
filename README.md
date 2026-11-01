# Customer Reviews Analysis: NLP, Time Series, and Neural Networks

Summative lab for an analytics firm client. The notebook analyzes Amazon customer reviews of e-reader devices (Nook and Kindle) in three parts: natural language processing, time series forecasting, and neural network rating prediction.

## Contents

| File | Description |
|------|-------------|
| `customer_reviews_analysis.ipynb` | Main notebook with all three parts and written reflections |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control (including the dataset) |

## Project Overview

### Part 1: NLP
- Cleans review text: lowercasing, URL and HTML removal, tokenization, stopword removal, and lemmatization.
- Finds the most frequent words and bigrams and visualizes them.
- **Finding:** Reviews mostly discuss reading devices and books ("nook", "book", "kindle") and features such as "battery life", "touch screen", and "customer service". Word frequency alone cannot show satisfaction, so it should be combined with ratings or sentiment.

### Part 2: Time Series
- Converts timestamps and resamples reviews to monthly counts and average ratings.
- Adds a 3-month rolling average of ratings.
- Tests stationarity with ADF and KPSS, using log and differencing transformations.
- Uses ACF/PACF plots to choose ARIMA orders, then compares candidate models against a naive forecast on a chronological train/test split.
- **Finding:** Reviews before 2010 are sparse, so modeling uses January 2010 to June 2014 (54 months). July 2014 is excluded because the data ends mid-month. Review volume grows over time, with bumps around the holiday periods.

### Part 3: Neural Networks
- Builds TF-IDF features (3,000 unigrams and bigrams) and trains a Keras multilayer perceptron to predict star ratings.
- Compares the network with a majority-class baseline and logistic regression.
- Tests variants: class weights, L2 regularization, dropout, early stopping, and different optimizers.
- Also tries a 3-class version (negative / neutral / positive).
- **Finding:** About 58% of test reviews are 5 stars, so accuracy is misleading and models are compared on macro F1 and per-class recall. The best network (test accuracy 0.570, macro F1 0.321) still overfit and added little over balanced logistic regression at this dataset size.

Each part ends with a written reflection that explains the results and their limitations.

## Dataset

The notebook expects a file named `amazon_reviews_lab.csv` in the same folder as the notebook, with at least these columns:

- `review_text`: the review text
- `rating`: star rating (1 to 5)
- `timestamp`: review date

The dataset is not included in this repository (it is listed in `.gitignore`). Add your copy before running the notebook.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/ainohamjeptoo-cloud/customer-reviews-analysis-lab.git
cd customer-reviews-analysis-lab

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add amazon_reviews_lab.csv to this folder, then launch Jupyter
jupyter notebook customer_reviews_analysis.ipynb
```

The first NLP cell downloads the required NLTK resources (punkt, stopwords, wordnet, omw-1.4) automatically. Run the notebook cells from top to bottom.

## Tools and Libraries

Python, pandas, NumPy, Matplotlib, NLTK, scikit-learn, statsmodels, TensorFlow/Keras, Jupyter.

## Author

Ainoham Jeptoo
