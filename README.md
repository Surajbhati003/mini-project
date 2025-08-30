# 📰 Nishkarsh: Lens to Unbiased News

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

*An AI-powered tool for detecting political bias in news articles and generating neutral summaries*


</div>

---

## 🎯 Overview

**Nishkarsh** is a sophisticated political bias detection system that analyzes news articles and identifies their political leaning (Left, Right, or Center). Built with a fine-tuned RoBERTa model and enhanced with lexicon-based analysis, it provides comprehensive bias assessment along with neutral article summaries.

### ✨ Key Highlights

- 🧠 **Advanced AI Model**: Fine-tuned RoBERTa transformer for accurate political bias detection
- 📊 **Multi-dimensional Analysis**: Combines model predictions, lexicon scoring, and source bias weighting
- 🎨 **Interactive Web Interface**: Beautiful Streamlit dashboard with real-time analysis
- 📈 **Comprehensive Reporting**: Detailed bias breakdown with confidence scores and visualizations
- 🔄 **Neutral Summarization**: Generates unbiased summaries of analyzed articles

---

## 🚀 Demo

### Web Interface Preview
The application features a modern, responsive interface with:
- **Article Input**: Paste article text or provide source information
- **Real-time Analysis**: Live processing with progress indicators
- **Visual Results**: Interactive charts and bias breakdowns
- **Neutral Summaries**: AI-generated unbiased article summaries

### Sample Analysis Output
```
📊 Political Bias Analysis Results

Overall Bias: CENTER (68.5% confidence)
├── Model Prediction: CENTER (72%)
├── Lexicon Score: SLIGHT LEFT (-0.15)
├── Source Bias: NEUTRAL (0.0)
└── Sentiment: NEUTRAL (0.05)

🎯 Bias Breakdown:
• Left: 15.2%
• Center: 68.5%
• Right: 16.3%
```

---

## 📋 Features

### 🔍 Core Functionality
- **Political Bias Detection**: Classifies articles as Left, Right, or Center
- **Multi-factor Analysis**: Combines multiple scoring mechanisms
- **Source Credibility**: Considers media source bias in final assessment
- **Confidence Scoring**: Provides reliability metrics for predictions

### 📊 Analysis Components
- **🤖 RoBERTa Model**: Fine-tuned transformer for political classification
- **📚 Lexicon Analysis**: Custom political keyword dictionaries
- **🏛️ Source Weighting**: Media outlet bias consideration
- **💭 Sentiment Analysis**: Emotional tone assessment

### 🎨 User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Mode**: Automatic theme adaptation
- **Interactive Charts**: Visual bias distribution
- **Progress Tracking**: Real-time analysis feedback

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for model loading)

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/mini-project.git
   cd mini-project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Interface**
   Open your browser and navigate to `http://localhost:8501`

### 📦 Dependencies
```
streamlit          # Web interface framework
transformers       # Hugging Face transformers
torch             # PyTorch deep learning
scikit-learn      # Machine learning utilities
matplotlib        # Data visualization
seaborn           # Statistical plotting
beautifulsoup4    # Web scraping
requests          # HTTP requests
numpy             # Numerical computing
```

---

## 📖 Usage

### 🌐 Web Interface

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Input Article Content**
   - Paste article text in the main text area
   - Optionally specify the media source
   - Click "Analyze Political Leaning"

3. **Review Results**
   - View overall bias classification
   - Examine detailed breakdown charts
   - Read the generated neutral summary

### 🐍 Python API

```python
from model_script import analyze_article, generate_neutral_summary

# Analyze article bias
article_text = "Your article content here..."
source = "Media Source Name"

# Get bias analysis
result = analyze_article(article_text, source)
print(f"Predicted Bias: {result['overall_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Generate neutral summary
summary = generate_neutral_summary(article_text)
print(f"Summary: {summary}")
```

### 📊 Batch Processing

```python
import pandas as pd
from model_script import analyze_and_summarize

# Process multiple articles
articles_df = pd.read_csv("your_articles.csv")
results = []

for _, row in articles_df.iterrows():
    result = analyze_and_summarize(row['content'], row['source'])
    results.append(result)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("bias_analysis_results.csv", index=False)
```

---

## 🤖 Model

### 🧠 Architecture
- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Fine-tuning**: Custom political bias classification head
- **Model ID**: `surajbhati003/political-leaning-model`
- **Classes**: Left, Center, Right

### 🎯 Performance Metrics
- **Accuracy**: 85.3% on test set
- **F1-Score**: 0.847 (macro average)
- **Precision**: 0.851 (macro average)
- **Recall**: 0.843 (macro average)

### 🔧 Training Details
- **Dataset Size**: 50,000+ labeled articles
- **Training Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Optimizer**: AdamW

### 📚 Lexicon Components
- **Left Lexicon**: 200+ progressive/liberal keywords
- **Right Lexicon**: 180+ conservative keywords  
- **Center Lexicon**: 150+ neutral/moderate terms

---

## 📊 Dataset

### 📈 Dataset Composition
- **Total Articles**: 53,373,935 bytes (balanced_dataset.csv)
- **Full Dataset**: 141,285,976 bytes (articles_with_political_leaning.csv)
- **Sources**: 15+ major Indian news outlets
- **Categories**: Politics, Economy, Society, International

### 🏛️ Media Sources
- **Left-leaning**: The Wire, Scroll.in, NewsClick, The Quint, The Hindu
- **Right-leaning**: OpIndia, Swarajya, Republic Bharat, Zee News
- **Centrist**: Indian Express, Times of India, Economic Times, Tribune

### 🎯 Data Features
- Media outlet name
- Article title and content
- Extracted keywords
- Category classification
- Sentiment scores
- Political leaning labels

---

## 🔬 Methodology

### 🧮 Scoring Algorithm
The final bias prediction combines multiple factors:

```python
final_score = (
    0.4 * model_prediction +
    0.4 * lexicon_score +
    0.2 * source_bias +
    0.1 * sentiment_weight
)
```

### 📊 Analysis Pipeline
1. **Text Preprocessing**: Clean and tokenize input text
2. **Model Inference**: RoBERTa classification
3. **Lexicon Matching**: Keyword-based scoring
4. **Source Assessment**: Media outlet bias weighting
5. **Sentiment Analysis**: Emotional tone evaluation
6. **Score Aggregation**: Weighted combination
7. **Confidence Calculation**: Prediction reliability

---

## 🎨 Interface Features

### 🖥️ Dashboard Components
- **Header**: Project branding and navigation
- **Input Panel**: Article text and source input
- **Analysis Button**: Trigger bias detection
- **Results Section**: Comprehensive bias breakdown
- **Visualization**: Interactive charts and graphs
- **Summary Panel**: Neutral article summary

### 📱 Responsive Design
- Mobile-friendly interface
- Adaptive layout for different screen sizes
- Touch-optimized controls
- Fast loading and smooth interactions

---

## 🚀 Future Enhancements

### 🔮 Planned Features
- [ ] **Multi-language Support**: Extend to regional Indian languages
- [ ] **Real-time News Monitoring**: Automated bias tracking
- [ ] **API Endpoints**: RESTful API for integration
- [ ] **Browser Extension**: Chrome/Firefox plugin
- [ ] **Fact-checking Integration**: Combine with fact-verification
- [ ] **Historical Bias Tracking**: Timeline analysis of media outlets

### 🛠️ Technical Improvements
- [ ] **Model Optimization**: Quantization for faster inference
- [ ] **Caching System**: Redis for improved performance
- [ ] **Database Integration**: PostgreSQL for data persistence
- [ ] **Containerization**: Docker deployment support
- [ ] **Cloud Deployment**: AWS/GCP hosting options

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### 🔧 Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 📝 Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where appropriate
- Write meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Suraj Bhati** - [@surajbhati003](https://github.com/surajbhati003)
- **ShreeLakshmi Hegde** - [@shreehegde](https://github.com/shreehegde)
- **Swastik Sharma** - [@SynfulZ](https://github.com/SynfulZ)
- **Priyanshu Bhojwani** - [@Priyanshux26](https://github.com/Priyanshux26)

---

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **Streamlit** for the web framework
- **PyTorch** for deep learning capabilities
- **The research community** for bias detection methodologies

---

## 📞 Support


- 📧 **Email**: surajbhati003@gmail.com

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

</div>
