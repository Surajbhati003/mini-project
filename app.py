import streamlit as st
from model_script import extract_article_info_fallback, analyze_and_summarize

st.set_page_config(page_title="News Bias & Summary", layout="centered")

st.title("📰 News Bias Detection & Neutral Summary")
st.markdown("Enter a news article URL to analyze its bias and get a neutral summary.")

url = st.text_input("Enter Link")
if url:
    with st.spinner("🔍 Extracting and analyzing article..."):
        article_info = extract_article_info_fallback(url)
        text = article_info["text"]
        source = article_info["source"]
        result, summary = analyze_and_summarize(text, source)

        st.subheader("🧠 Final Political Leaning:")
        st.success(f"**{result['Final Conclusion']}**")

        st.subheader("📊 Detailed Bias Scores")
        st.markdown(f"""
        - **TF-IDF Scores**  
          Left: `{result['TFIDF_Scores'][0]:.4f}`  
          Right: `{result['TFIDF_Scores'][1]:.4f}`  
          Centrist: `{result['TFIDF_Scores'][2]:.4f}`

        - **RoBERTa Embedding Similarities**  
          Left: `{result['RoBERTa_Similarities'][0]:.4f}`  
          Right: `{result['RoBERTa_Similarities'][1]:.4f}`  
          Centrist: `{result['RoBERTa_Similarities'][2]:.4f}`

        - **Source Weight (based on domain)**  
          Left: `{result['Source_Weights'][0]:.2f}`  
          Right: `{result['Source_Weights'][1]:.2f}`  
          Centrist: `{result['Source_Weights'][2]:.2f}`
        """)

        st.subheader("📝 Neutral Summary")
        st.info(summary)
