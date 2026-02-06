# Toxic Comment Analyzer - Project Notes

## Quick Elevator Pitch
"I built a machine learning model that detects toxic comments, then created an interactive dashboard to analyze where and why the model makes mistakes. This is directly relevant to AI safety - understanding model failures helps build better content moderation systems."

---

## What This Project Does

### The Problem
AI companies use machine learning models to automatically detect toxic content (hate speech, insults, threats, etc.) on their platforms. But these models make mistakes:
- **False Positives**: Flagging legitimate comments as toxic (over-censorship)
- **False Negatives**: Missing actually toxic content (harmful content gets through)

Understanding these errors is critical for improving AI safety.

### My Solution
1. **Train a classifier** to detect toxic comments
2. **Analyze the errors** to find patterns in where it fails
3. **Build a dashboard** to explore these errors interactively

---

## The Dataset

**Source**: Civil Comments dataset from Hugging Face (originally from Google/Jigsaw)

**Size**: ~1.8 million comments (I used 50,000 for this project)

**Labels**: Each comment has a toxicity score from 0.0 to 1.0, plus subcategory scores:
- `toxicity` - overall toxicity (main label)
- `severe_toxicity` - extremely toxic
- `obscene` - contains obscene language
- `threat` - contains threats
- `insult` - contains insults
- `identity_attack` - attacks based on identity (race, gender, etc.)
- `sexual_explicit` - sexually explicit content

**Class Imbalance**: Only ~6.6% of comments are toxic (score >= 0.5). This is realistic - most online comments are fine. But it makes the classification harder.

---

## The Model

### Why I Chose This Approach

**TF-IDF + Logistic Regression** (not deep learning)

Reasons:
1. **Interpretable** - I can understand WHY the model makes predictions
2. **Fast to train** - minutes, not hours
3. **Good baseline** - industry-standard for text classification
4. **Focus on analysis** - this project is about error analysis, not building the best model

### How TF-IDF Works
TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numbers:
- Words that appear often in a document get higher weight
- Words that appear in EVERY document get lower weight (like "the", "and")
- Result: Each comment becomes a vector of ~10,000 numbers

### How Logistic Regression Works
- Takes the TF-IDF vector as input
- Learns which words correlate with toxicity
- Outputs a probability (0 to 1) that the comment is toxic
- If probability >= 0.5, classify as toxic

### Model Performance
```
Accuracy: 89%
Precision (Toxic): 34%  <- When it says toxic, it's right 34% of the time
Recall (Toxic): 66%     <- It catches 66% of actual toxic comments
```

**Why is precision so low?**
I used `class_weight='balanced'` which makes the model more aggressive at catching toxic content. This means more false positives (over-censorship) but fewer false negatives (missed toxicity). This is a common trade-off in content moderation.

---

## Dashboard Sections Explained

### 1. Overview (Top Metrics)

| Metric | What It Means |
|--------|---------------|
| Showing X of Y total | How many comments match your current filters |
| Accuracy | Percentage of correct predictions overall |
| Correct | Number of predictions that were right |
| False Positives | Non-toxic comments incorrectly flagged as toxic |
| False Negatives | Toxic comments the model missed |

**Key insight**: In content moderation, false positives = censorship of legitimate speech, false negatives = harmful content getting through. Both are bad but in different ways.

### 2. Error Distribution

**Confusion Matrix**
A 2x2 grid showing:
- Top-left: True Negatives (correctly identified as non-toxic)
- Top-right: False Positives (wrongly flagged as toxic)
- Bottom-left: False Negatives (toxic content missed)
- Bottom-right: True Positives (correctly identified as toxic)

**Prediction Distribution Pie Chart**
Visual breakdown of Correct vs False Positive vs False Negative predictions.

### 3. Confidence Analysis

**Confidence Distribution by Error Type (Box Plot)**
Shows how confident the model was for each type of prediction:
- **Correct predictions**: Model is confident and right
- **False Positives**: Model was confident it was toxic, but wrong
- **False Negatives**: Model wasn't confident enough, so it missed the toxicity

**Key insight**: Look for "overconfident errors" - cases where the model was very sure (>80% confidence) but completely wrong. These are the most concerning failures.

**Overconfident Errors**
Count of predictions where model was >80% confident but wrong. These deserve special attention because the model had no idea it was wrong.

### 4. Error Patterns by Comment Length

**Error Rate by Comment Length**
Bar chart showing: Do shorter or longer comments cause more errors?

**Error Type by Comment Length**
Grouped bar chart showing false positives vs false negatives at each length.

**Key insight**: Models often struggle with very short comments (not enough context) or very long comments (more complexity).

### 5. Error Patterns by Toxicity Type

**False Negative Rate by Subcategory**
Which types of toxic content does the model miss most often?
- High bar = model is bad at catching this type
- Example: If "Threat" has high miss rate, the model is bad at detecting threats

**Subcategory Distribution in Errors**
Of the toxic comments we missed, what types were they?
- Helps prioritize what to improve

**Key insight**: If the model misses most "identity_attack" comments, that's a bias/fairness concern. If it misses "threats", that's a safety concern.

### 6. Explore Individual Comments

Interactive section to read actual misclassified comments:
- Filter by error type, confidence, length
- See the actual text
- Understand WHY the model might have failed

**Key insight**: Reading actual errors often reveals patterns that statistics miss. Maybe the model fails on sarcasm, or specific slang, or context-dependent toxicity.

---

## Key Talking Points for Interviews

### Why This Project Matters
"AI safety is one of the biggest challenges in the industry. Models that moderate content affect millions of people. Understanding where they fail - and who they fail on - is critical for building fair, effective systems."

### Technical Skills Demonstrated
- **NLP/Text Processing**: TF-IDF vectorization, text classification
- **Machine Learning**: Logistic Regression, handling imbalanced data
- **Data Analysis**: Error analysis, pattern recognition
- **Visualization**: Plotly, interactive dashboards
- **Deployment**: Streamlit Cloud, GitHub

### What I Learned
1. Class imbalance is a real challenge - only 6.6% of comments were toxic
2. There's always a trade-off between false positives and false negatives
3. Different types of toxicity (threats vs insults vs obscenity) have different detection rates
4. Reading individual errors reveals patterns that aggregate statistics miss

### If I Had More Time, I Would...
1. Add word-level analysis (which words cause the most errors?)
2. Test for demographic bias (does it fail more on certain dialects?)
3. Try more advanced models (BERT, etc.) and compare error patterns
4. Add a feature to test new comments in real-time

---

## Files in This Project

```
Toxic-Comment-Analyzer/
├── app.py                 # Streamlit dashboard (main app)
├── requirements.txt       # Python dependencies
├── README.md              # Project overview for GitHub
├── .gitignore             # Files to exclude from git
├── src/
│   └── train_model.py     # Script to train the model
├── data/
│   ├── predictions.csv    # Model predictions for error analysis
│   ├── model.pkl          # Trained model (not in git)
│   └── vectorizer.pkl     # TF-IDF vectorizer (not in git)
├── notes/
│   └── Project-Overview.md  # This file
└── venv/                  # Python virtual environment (not in git)
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/cblythe8/Toxic-Comment-Analyzer.git
cd Toxic-Comment-Analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model
pip install datasets scikit-learn
python src/train_model.py

# 5. Run the dashboard
streamlit run app.py
```

---

## Links

- **Live Demo**: https://toxic-comment-analyzer-newyxkfu6fydsvg5jf8qnz.streamlit.app
- **GitHub Repo**: https://github.com/cblythe8/Toxic-Comment-Analyzer
- **Dataset**: https://huggingface.co/datasets/google/civil_comments
