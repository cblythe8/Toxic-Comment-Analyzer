# Toxic Comment Detection - Error Analysis Dashboard

A portfolio project demonstrating ML error analysis skills for data analyst roles at AI companies.

## Project Overview

Train a text classifier to detect toxic comments, then build an interactive dashboard to analyze where and why the model fails.

## Why This Matters

AI safety is a critical concern. Understanding model errors helps:
- Reduce false positives (over-censorship of legitimate speech)
- Reduce false negatives (harmful content slipping through)
- Identify bias in model predictions

## Tech Stack

- **Data Processing:** pandas, numpy
- **NLP:** scikit-learn (TF-IDF), NLTK
- **Modeling:** Logistic Regression
- **Visualization:** matplotlib, plotly
- **Dashboard:** Streamlit

## Dataset

[Civil Comments](https://huggingface.co/datasets/google/civil_comments) from Hugging Face (~1.8M labeled comments)

## Project Structure

```
Toxic-Comment-Analyzer/
├── data/               # Dataset files (not committed to git)
├── src/                # Python source files
├── notebooks/          # Jupyter notebooks for exploration
├── app.py              # Streamlit dashboard
├── requirements.txt    # Dependencies
└── README.md
```

## Error Analysis Components

- Confusion matrix & classification metrics
- False positive analysis (legit comments flagged as toxic)
- False negative analysis (toxic comments missed)
- Confidence score distribution on errors
- Word/phrase correlation with misclassifications
- Error patterns by comment length
- Performance breakdown by toxicity subcategory

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

[View Dashboard](https://toxic-comment-analyzer-newyxkfu6fydsvg5jf8qnz.streamlit.app)

## Author

Colin Blythe - [GitHub](https://github.com/cblythe8)
