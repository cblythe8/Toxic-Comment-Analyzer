"""
Toxic Comment Detection - Error Analysis Dashboard
Explore where and why the model makes mistakes.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Toxic Comment Error Analysis",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load predictions data."""
    df = pd.read_csv('data/predictions.csv')

    # Add error type column
    df['error_type'] = 'Correct'
    df.loc[(df['actual'] == 0) & (df['predicted'] == 1), 'error_type'] = 'False Positive'
    df.loc[(df['actual'] == 1) & (df['predicted'] == 0), 'error_type'] = 'False Negative'

    # Add length bins
    df['length_bin'] = pd.cut(
        df['text_length'],
        bins=[0, 50, 100, 200, 500, 1000, 2000],
        labels=['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    )

    # Add subcategory flags
    subcategories = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
    for cat in subcategories:
        df[f'{cat}_flag'] = (df[cat] >= 0.5).astype(int)

    return df

# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Data not found. Run `python src/train_model.py` first.")
    st.stop()

# Sidebar
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("All charts update based on these filters.")
st.sidebar.markdown("---")

# Filters
error_filter = st.sidebar.multiselect(
    "Error Type",
    options=['Correct', 'False Positive', 'False Negative'],
    default=['Correct', 'False Positive', 'False Negative']
)

confidence_range = st.sidebar.slider(
    "Confidence Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05
)

length_filter = st.sidebar.multiselect(
    "Comment Length",
    options=['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+'],
    default=['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
)

# Apply filters
filtered_df = df[
    (df['error_type'].isin(error_filter)) &
    (df['confidence'] >= confidence_range[0]) &
    (df['confidence'] <= confidence_range[1]) &
    (df['length_bin'].isin(length_filter))
]

# Main content
st.title("Toxic Comment Detection - Error Analysis Dashboard")
st.markdown("Analyze where and why the model makes mistakes to improve AI safety.")

# Overview metrics - show both total and filtered
st.header("📊 Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# Filtered stats
filtered_total = len(filtered_df)
filtered_correct = len(filtered_df[filtered_df['correct'] == 1])
filtered_fp = len(filtered_df[filtered_df['error_type'] == 'False Positive'])
filtered_fn = len(filtered_df[filtered_df['error_type'] == 'False Negative'])
filtered_accuracy = (filtered_correct / filtered_total * 100) if filtered_total > 0 else 0

col1.metric("Showing", f"{filtered_total:,}", f"of {len(df):,} total")
col2.metric("Accuracy", f"{filtered_accuracy:.1f}%")
col3.metric("Correct", f"{filtered_correct:,}")
col4.metric("False Positives", f"{filtered_fp:,}", help="Non-toxic flagged as toxic (over-censorship)")
col5.metric("False Negatives", f"{filtered_fn:,}", help="Toxic missed (harm gets through)")

st.markdown("---")

# Check if we have data to show
if filtered_total == 0:
    st.warning("No data matches your current filters. Adjust the sidebar filters.")
    st.stop()

# Confusion Matrix and Error Distribution
st.header("📈 Error Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")

    # Create confusion matrix from filtered data
    cm_values = [
        [len(filtered_df[(filtered_df['actual'] == 0) & (filtered_df['predicted'] == 0)]),
         len(filtered_df[(filtered_df['actual'] == 0) & (filtered_df['predicted'] == 1)])],
        [len(filtered_df[(filtered_df['actual'] == 1) & (filtered_df['predicted'] == 0)]),
         len(filtered_df[(filtered_df['actual'] == 1) & (filtered_df['predicted'] == 1)])]
    ]

    fig_cm = px.imshow(
        cm_values,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Non-toxic', 'Toxic'],
        y=['Non-toxic', 'Toxic'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("Prediction Distribution")

    error_counts = filtered_df['error_type'].value_counts()
    colors = {'Correct': '#2ecc71', 'False Positive': '#e74c3c', 'False Negative': '#f39c12'}

    fig_pie = px.pie(
        values=error_counts.values,
        names=error_counts.index,
        color=error_counts.index,
        color_discrete_map=colors
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# Confidence Analysis
st.header("🎯 Confidence Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confidence Distribution by Error Type")

    fig_conf = px.box(
        filtered_df,
        x='error_type',
        y='confidence',
        color='error_type',
        color_discrete_map={'Correct': '#2ecc71', 'False Positive': '#e74c3c', 'False Negative': '#f39c12'},
        category_orders={'error_type': ['Correct', 'False Positive', 'False Negative']}
    )
    fig_conf.update_layout(
        xaxis_title="Error Type",
        yaxis_title="Model Confidence (probability of toxic)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_conf, use_container_width=True)

with col2:
    st.subheader("Overconfident Errors")

    # Find errors where model was very confident but wrong
    errors = filtered_df[filtered_df['correct'] == 0].copy()
    if len(errors) > 0:
        errors['overconfidence'] = errors.apply(
            lambda x: x['confidence'] if x['predicted'] == 1 else 1 - x['confidence'],
            axis=1
        )
        overconfident = errors[errors['overconfidence'] > 0.8]
        st.metric("High Confidence Errors (>80%)", len(overconfident))
        st.markdown("*These are cases where the model was very sure but completely wrong.*")

        if len(overconfident) > 0:
            breakdown = overconfident['error_type'].value_counts()
            st.dataframe(breakdown.to_frame('Count'))
        else:
            st.info("No high-confidence errors in current filter.")
    else:
        st.metric("High Confidence Errors (>80%)", 0)
        st.info("No errors in current filter.")

st.markdown("---")

# Comment Length Analysis
st.header("📏 Error Patterns by Comment Length")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Error Rate by Comment Length")

    length_errors = filtered_df.groupby('length_bin', observed=True).agg({
        'correct': ['sum', 'count']
    }).reset_index()
    length_errors.columns = ['length_bin', 'correct_count', 'total']
    length_errors['error_rate'] = (1 - length_errors['correct_count'] / length_errors['total']) * 100

    fig_len = px.bar(
        length_errors,
        x='length_bin',
        y='error_rate',
        color='error_rate',
        color_continuous_scale='Reds'
    )
    fig_len.update_layout(
        xaxis_title="Comment Length (characters)",
        yaxis_title="Error Rate (%)",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_len, use_container_width=True)

with col2:
    st.subheader("Error Type by Comment Length")

    length_breakdown = filtered_df.groupby(['length_bin', 'error_type'], observed=True).size().reset_index(name='count')

    fig_len_type = px.bar(
        length_breakdown,
        x='length_bin',
        y='count',
        color='error_type',
        barmode='group',
        color_discrete_map={'Correct': '#2ecc71', 'False Positive': '#e74c3c', 'False Negative': '#f39c12'}
    )
    fig_len_type.update_layout(
        xaxis_title="Comment Length (characters)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_len_type, use_container_width=True)

st.markdown("---")

# Subcategory Analysis
st.header("🏷️ Error Patterns by Toxicity Type")

subcategories = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

col1, col2 = st.columns(2)

with col1:
    st.subheader("False Negative Rate by Subcategory")
    st.markdown("*What types of toxic content does the model miss most often?*")

    fn_rates = []
    for cat in subcategories:
        subset = filtered_df[filtered_df[f'{cat}_flag'] == 1]
        if len(subset) > 0:
            fn_rate = len(subset[subset['error_type'] == 'False Negative']) / len(subset) * 100
            fn_rates.append({'category': cat.replace('_', ' ').title(), 'fn_rate': fn_rate, 'count': len(subset)})

    if len(fn_rates) > 0:
        fn_df = pd.DataFrame(fn_rates)
        fig_fn = px.bar(
            fn_df,
            x='category',
            y='fn_rate',
            color='fn_rate',
            color_continuous_scale='Oranges',
            hover_data=['count']
        )
        fig_fn.update_layout(
            xaxis_title="Toxicity Subcategory",
            yaxis_title="Miss Rate (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_fn, use_container_width=True)
    else:
        st.info("No toxic subcategory data in current filter.")

with col2:
    st.subheader("Subcategory Distribution in Errors")

    # Count subcategories in false negatives
    errors = filtered_df[filtered_df['error_type'] == 'False Negative']
    subcat_counts = []
    for cat in subcategories:
        count = errors[f'{cat}_flag'].sum() if len(errors) > 0 else 0
        subcat_counts.append({'category': cat.replace('_', ' ').title(), 'count': count})

    subcat_df = pd.DataFrame(subcat_counts)

    fig_subcat = px.bar(
        subcat_df,
        x='category',
        y='count',
        color='count',
        color_continuous_scale='Reds'
    )
    fig_subcat.update_layout(
        xaxis_title="Toxicity Subcategory",
        yaxis_title="Count in Missed Comments",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_subcat, use_container_width=True)

st.markdown("---")

# Explore Individual Errors
st.header("🔎 Explore Individual Comments")

st.markdown(f"**Showing {len(filtered_df)} comments** based on your filters")

# Sort options
col1, col2 = st.columns(2)
with col1:
    sort_by = st.selectbox(
        "Sort by",
        options=['confidence', 'text_length', 'toxicity'],
        index=0
    )
with col2:
    sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True)

sorted_df = filtered_df.sort_values(
    sort_by,
    ascending=(sort_order == 'Ascending')
)

# Display comments
for idx, row in sorted_df.head(20).iterrows():
    with st.expander(f"{row['error_type']} | Confidence: {row['confidence']:.2f} | Length: {row['text_length']}"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Comment:**")
            st.markdown(f"> {row['text']}")

        with col2:
            st.markdown("**Details:**")
            st.write(f"Actual: {'Toxic' if row['actual'] == 1 else 'Non-toxic'}")
            st.write(f"Predicted: {'Toxic' if row['predicted'] == 1 else 'Non-toxic'}")
            st.write(f"Toxicity Score: {row['toxicity']:.2f}")

            # Show subcategories if toxic
            if row['actual'] == 1:
                st.markdown("**Subcategories:**")
                for cat in subcategories:
                    if row[cat] >= 0.5:
                        st.write(f"• {cat.replace('_', ' ').title()}")

st.markdown("---")

# Footer
st.markdown("""
### About This Dashboard

This dashboard analyzes errors from a toxic comment classifier to understand:
- **False Positives**: Non-toxic comments incorrectly flagged (over-censorship)
- **False Negatives**: Toxic comments that slip through (safety risk)

Understanding these errors is critical for AI safety and content moderation.

**Model**: TF-IDF + Logistic Regression
**Dataset**: Civil Comments (50k samples)

Built by Colin Blythe | [GitHub](https://github.com/cblythe8)
""")
