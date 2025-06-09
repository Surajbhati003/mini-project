# app.py
import streamlit as st
from model_script import analyze_article, extract_article_info_fallback, generate_neutral_summary
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

# Configure page
st.set_page_config(
    page_title="Nishkarsh: Lens to unbiased News",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - fixed for both light and dark modes
st.markdown("""
<style>
    :root {
        --primary: #4e73df;
        --danger: #e74a3b;
        --success: #1cc88a;
        --info: #36b9cc;
        --card-bg: #ffffff;
        --text-color: #2e3a59;
    }
    
    [data-theme="dark"] {
        --primary: #4e73df;
        --danger: #e74a3b;
        --success: #1cc88a;
        --info: #36b9cc;
        --card-bg: #1e2130;
        --text-color: #f0f2f6;
    }
    
    body {
        color: var(--text-color);
        background-color: var(--card-bg);
    }
    
    .stProgress > div > div > div > div {
        background-color: var(--primary);
    }
    
    .metric-card {
        background: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,.1);
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(0,0,0,.1);
    }
    
    .metric-card h4 {
        color: var(--text-color);
        font-size: 1rem;
        margin-bottom: 10px;
    }
    
    .metric-card h2 {
        color: var(--text-color);
        font-size: 1.8rem;
        margin: 0;
    }
    
    .leaning-label {
        font-weight: 700;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.4rem;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,.1);
    }
    
    .left-leaning {
        background-color: var(--danger);
        color: white;
    }
    
    .right-leaning {
        background-color: var(--success);
        color: white;
    }
    
    .center-leaning {
        background-color: var(--info);
        color: white;
    }
    
    .analysis-section {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(0,0,0,.1);
        box-shadow: 0 4px 6px rgba(0,0,0,.1);
    }
    
    .summary-box {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(0,0,0,.1);
        box-shadow: 0 4px 6px rgba(0,0,0,.1);
    }
    
    .stTextArea textarea {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
    
    .stTextInput input {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("📰 Nishkarsh: Lens to unbiased News")
st.markdown("""
<div style="font-size: 1.1rem; line-height: 1.6;">
Analyze news articles for political bias using advanced ML techniques. 
This tool evaluates content for left-leaning, right-leaning, or centrist perspectives.
</div>
""", unsafe_allow_html=True)

# Input Section
with st.expander("📥 INPUT ARTICLE CONTENT", expanded=True):
    input_method = st.radio("Select input method:", ("Enter URL", "Paste Text"), horizontal=True)
    
    article_text = ""
    source = ""
    
    if input_method == "Enter URL":
        url = st.text_input("Article URL:", placeholder="https://example.com/news-article")
        if url:
            with st.spinner("Extracting article content..."):
                try:
                    article_info = extract_article_info_fallback(url)
                    article_text = article_info['text']
                    source = article_info['source']
                    st.success(f"✅ Extracted content from: {source}")
                except Exception as e:
                    st.error(f"❌ Error extracting content: {str(e)}")
    
    else:  # Paste Text
        article_text = st.text_area("Article Text:", height=250, 
                                   placeholder="Paste the full article text here...")
        source = st.text_input("News Source (optional):", placeholder="e.g., The Times of India")

# Analysis Section
if st.button("Analyze Political Leaning", type="primary", use_container_width=True):
    if not article_text.strip():
        st.warning("⚠️ Please provide article content to analyze")
        st.stop()
    
    with st.spinner("Analyzing content. This may take a moment..."):
        # Add artificial delay to show progress
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        try:
            # Perform analysis
            bias_result = analyze_article(article_text, source)
            summary = generate_neutral_summary(article_text)
            
            # Display Results
            st.subheader("📊 Analysis Results")
            
            # Leaning Conclusion and All Graphs in a Row
            st.markdown("### Final Conclusion")
            leaning = bias_result['Final Conclusion']
            if leaning == "Leftist":
                st.markdown(f'<div class="leaning-label left-leaning">LEFT-LEANING</div>', unsafe_allow_html=True)
            elif leaning == "Rightist":
                st.markdown(f'<div class="leaning-label right-leaning">RIGHT-LEANING</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="leaning-label center-leaning">CENTRIST</div>', unsafe_allow_html=True)

            # Prepare data for all plots
            scores = bias_result['Final_Scores']
            categories = list(scores.keys())
            values = list(scores.values())
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            values_plot = values + values[:1]

            tfidf_data = {
                "Leftist": bias_result['TFIDF_Scores'][0],
                "Centrist": bias_result['TFIDF_Scores'][2],
                "Rightist": bias_result['TFIDF_Scores'][1]
            }

            sim_data = {
                "Leftist": bias_result['RoBERTa_Similarities'][0],
                "Centrist": bias_result['RoBERTa_Similarities'][2],
                "Rightist": bias_result['RoBERTa_Similarities'][1]
            }

            colors = ["#e74a3b", "#36b9cc", "#1cc88a"]

            # Create columns for the three plots
            col1, col2, col3 = st.columns(3)

            with col1:
                # Radar Chart
                fig1, ax1 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                plt.xticks(angles[:-1], categories, color='grey', size=10)
                ax1.set_rlabel_position(0)
                max_val = max(values) * 1.2
                plt.yticks([max_val/4, max_val/2, max_val*0.75], 
                           [f"{max_val/4:.1f}", f"{max_val/2:.1f}", f"{max_val*0.75:.1f}"], 
                           color="grey", size=8)
                plt.ylim(0, max_val)
                ax1.plot(angles, values_plot, linewidth=1, linestyle='solid')
                ax1.fill(angles, values_plot, alpha=0.2)
                plt.title('Political Leaning Score', size=14, color='black', y=1.1)
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                # Horizontal Bar Chart (TF-IDF)
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                sns.barplot(
                    x=list(tfidf_data.values()), 
                    y=list(tfidf_data.keys()),
                    palette=colors,
                    orient='h',
                    ax=ax2
                )
                ax2.set_xlabel("TF-IDF Score")
                ax2.set_title("Political Term Frequency Analysis")
                st.pyplot(fig2)
                plt.close(fig2)

            with col3:
                # Pie Chart (Similarity)
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                explode = (0.05, 0.05, 0.05)
                wedges, texts, autotexts = ax3.pie(
                    sim_data.values(), 
                    labels=sim_data.keys(), 
                    autopct='%1.1f%%',
                    colors=colors,
                    explode=explode,
                    startangle=90,
                    shadow=True
                )
                ax3.axis('equal')
                plt.setp(autotexts, size=12, weight="bold", color="white")
                ax3.set_title("Embedding Similarity Distribution")
                st.pyplot(fig3)
                plt.close(fig3)

            # Detailed Metrics
            st.subheader("🔍 Detailed Analysis")
            
            cols = st.columns(3)
            metrics = [
                ("Model Confidence", f"{bias_result['Model_Prediction'][1]*100:.1f}%", "#4e73df"),
                ("Sentiment Influence", f"{bias_result['Sentiment_Score']:.2f}", "#6f42c1"),
                ("Source Influence", f"{max(bias_result['Source_Weights']):.2f}", "#fd7e14")
            ]
            
            for col, metric in zip(cols, metrics):
                with col:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<h4>{metric[0]}</h4>"
                        f"<h2 style='color: {metric[2]}'>{metric[1]}</h2>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
            
            # Neutral Summary
            st.subheader("📝 Neutral Summary")
            with st.container():
                st.markdown(
                    f'<div class="summary-box" style="color: black;">{summary}</div>',
                    unsafe_allow_html=True
                )
            
            # Raw Data (optional)
            with st.expander("📄 Raw Analysis Data"):
                st.json(bias_result)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

# Sidebar with information
st.sidebar.header("About This Tool")
st.sidebar.markdown("""
This Nishkarsh: Lens to unbiased News uses:
- **RoBERTa model** fine-tuned for political detection
- **Lexicon analysis** with custom political dictionaries
- **TF-IDF scoring** of political terminology
- **Source reputation** weighting
- **Sentiment analysis** for bias detection
""")

st.sidebar.subheader("Recognized Sources")
left, center, right = st.sidebar.columns(3)

with left:
    st.markdown("**Left-leaning**")
    st.caption("The Wire, Scroll.in, NewsClick, The Quint, The Hindu")

with center:
    st.markdown("**Centrist**")
    st.caption("Indian Express, Times of India, Economic Times, Tribune")

with right:
    st.markdown("**Right-leaning**")
    st.caption("OpIndia, Swarajya, Republic Bharat, Zee News, NDTV, ABP")

st.sidebar.markdown("---")
st.sidebar.caption("Nishkarsh: Lens to unbiased News v1.0 | © 2023")