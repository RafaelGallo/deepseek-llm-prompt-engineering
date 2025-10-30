# 🤖 **DeepSeek Sentiment Analysis & Keyword Extraction**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/Transformers-DeepSeek_7B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-🤗_Transformers-yellow)
![Pandas](https://img.shields.io/badge/pandas-Data_Analysis-green)
![NumPy](https://img.shields.io/badge/NumPy-Linear_Algebra-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-lightgreen)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-cyan)
![WordCloud](https://img.shields.io/badge/WordCloud-Text_Visualization-lightgrey)
![tqdm](https://img.shields.io/badge/tqdm-Progress_Bars-purple)
![GPU](https://img.shields.io/badge/Accelerated-NVIDIA_GPU-success)
![Status](https://img.shields.io/badge/Status-Production-success)
![License](https://img.shields.io/badge/License-MIT-black)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Kaggle_&_Colab-blue)
![Author](https://img.shields.io/badge/Author-Mago_Dados_(Rafael_Gallo)-red)

<p align="center">
  <img src="https://github.com/RafaelGallo/deepseek-llm-prompt-engineering/blob/main/img/logo.png?raw=true" width="70%" alt="DeepSeek Sentiment Logo"/>
</p>

## 🚀 **Introduction**

The **DeepSeek Sentiment Analysis & Keyword Extraction** project is an advanced NLP pipeline that combines **Large Language Models (LLMs)** with **classical machine learning** to deliver high-quality sentiment classification and keyword discovery from real-world social media data.

Using the **DeepSeek-7B** model as a reasoning core, this project leverages **prompt engineering**, **few-shot learning**, and **embedding extraction** to understand context, tone, and intent in tweets.
From this foundation, multiple **supervised machine learning algorithms** (Logistic Regression, SVM, RandomForest, XGBoost, and more) are trained on DeepSeek’s semantic embeddings, enabling robust predictions with explainable performance metrics.

Beyond classification, the system also employs **LLM-driven keyword extraction**, allowing it to automatically identify the **most relevant entities, emotions, and topics** mentioned in each tweet — turning unstructured text into structured, actionable insights.

This hybrid architecture bridges **LLM intelligence** and **traditional predictive analytics**, delivering a scalable solution for **brand monitoring, customer feedback analysis, and social media sentiment tracking**.

### 💡 In short:

> “DeepSeek analyzes what people *say* — and uncovers *why* they say it.”

## 🧩 **Pipeline Summary**

| Step                       | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| 🧹 **Preprocessing**       | Cleaning and normalizing tweet text (lowercase, remove stopwords, etc.) |
| 🧠 **LLM Prompting**       | Sentiment classification using DeepSeek with few-shot prompts           |
| 🗝️ **Keyword Extraction** | Using prompt engineering to extract 5–10 main keywords per tweet        |
| 🔡 **Embeddings**          | Generating dense embeddings from DeepSeek last hidden layer             |
| ⚙️ **Machine Learning**    | Training 9 models (Logistic Regression, SVM, RF, XGB, etc.)             |
| 📈 **Evaluation**          | Metrics: Accuracy, F1-Score, Recall, Precision, ROC-AUC                 |
| ☁️ **Visualization**       | WordClouds by sentiment + Feature Importance + ROC Curves               |

## 🧠 **Technologies Used**

* **LLM:** DeepSeek 7B (Hugging Face)
* **Frameworks:** PyTorch, Transformers, scikit-learn
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **Optimization:** tqdm (batch progress), BitsAndBytes quantization
* **Environment:** GPU (CUDA)

## 📊 **Dataset**

**Twitter Sentiment Analysis Dataset**

* **Columns:** `Tweet ID`, `Entity`, `Sentiment`, `Tweet Content`
* **Classes:** `Positive`, `Negative`, `Neutral`
* **Language:** English

Example:

| Tweet                                   | Entity    | Sentiment |
| --------------------------------------- | --------- | --------- |
| "Apple's new iPhone is amazing!"        | Apple     | Positive  |
| "Microsoft's update is full of bugs."   | Microsoft | Negative  |
| "Samsung launched another phone today." | Samsung   | Neutral   |

## 🧱 **Project Structure**

```
deepseek_sentiment_analysis/
│
├── data/
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
│
├── models/
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   ├── LightGBM.pkl
│   ├── XGBoost.pkl
│
├── notebooks/
│   └── 01_deepseek_sentiment.ipynb
│
├── app/
│   ├── inference.py
│   ├── generate_embeddings.py
│   ├── evaluate_models.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🧠 **Prompt Engineering Example**

```python
FEW_SHOT = """
Examples:
Text: "Apple's campaign is amazing, I loved the innovation!"
Label: Positive

Text: "Microsoft's new update is terrible, full of bugs."
Label: Negative

Text: "Samsung's new ad was okay, nothing special."
Label: Neutral

Instructions:
Classify the sentiment of the following text.
Answer ONLY with one of the three words:
Positive, Negative, or Neutral.
"""

def make_prompt(text, camp):
    text = str(text).strip().replace("\n", " ")
    return f"{FEW_SHOT}\nCampaign: {camp}\nText: \"{text}\"\nLabel:"
```

## 🗝️ **Keyword Extraction Prompt**

```python
KEYWORD_PROMPT = """
You are an NLP assistant specialized in keyword extraction.

Instructions:
Given a text about a campaign or opinion, extract 5 to 10 of the most relevant keywords.
Return ONLY a comma-separated list, with no explanations.

Examples:

Text: "Apple's new iPhone release was amazing, with a beautiful design and great battery life!"
Keywords: iPhone, Apple, release, design, battery life, amazing
"""
```

## 📈 **Model Evaluation**

| Model              | Accuracy | F1-Score | Recall | AUC-Macro |
| ------------------ | -------- | -------- | ------ | --------- |
| RandomForest       | 0.845    | 0.843    | 0.847  | 0.882     |
| XGBoost            | 0.839    | 0.836    | 0.839  | 0.877     |
| LightGBM           | 0.833    | 0.830    | 0.832  | 0.870     |
| GradientBoosting   | 0.823    | 0.820    | 0.823  | 0.861     |
| LogisticRegression | 0.811    | 0.808    | 0.812  | 0.852     |
| SVM                | 0.790    | 0.782    | 0.785  | —         |
| KNN                | 0.771    | 0.761    | 0.763  | —         |
| DecisionTree       | 0.763    | 0.755    | 0.758  | 0.801     |
| AdaBoost           | 0.755    | 0.748    | 0.750  | 0.812     |

## ☁️ **WordCloud Visualization**

Each sentiment generates a separate cloud highlighting the most frequent terms:

| Sentiment | Example                           |
| --------- | --------------------------------- |
| Positive  | love, amazing, design, innovation |
| Negative  | bug, crash, slow, terrible        |
| Neutral   | event, release, update, launch    |

## 📦 **Installation**

```bash
git clone https://github.com/SEU_USUARIO/deepseek-sentiment-analysis.git
cd deepseek-sentiment-analysis
pip install -r requirements.txt
```

## 🧩 **Run Notebook**

```bash
jupyter notebook notebooks/01_deepseek_sentiment.ipynb
```

Or run via script:

```bash
python app/inference.py
```

## 🏗️ **Requirements**

```txt
transformers>=4.45.0
torch>=2.4.0
tqdm
pandas
scikit-learn
matplotlib
seaborn
wordcloud
xgboost
lightgbm
```

## 🧠 **Example Output (DeepSeek Keyword Extraction)**

| clean_text                                | Predicted | keywords                                           |
| ----------------------------------------- | --------- | -------------------------------------------------- |
| apple release amazing design battery life | Positive  | Apple, release, design, battery life, amazing      |
| microsoft update slow bugs disappointed   | Negative  | Microsoft, update, bugs, performance, disappointed |
| samsung event normal product              | Neutral   | Samsung, event, product, campaign                  |

## 🧩 **Results**

✅ DeepSeek LLM classification reached **F1 = 0.84**

✅ ML ensemble (RandomForest/XGBoost) achieved **AUC = 0.88**

✅ Keyword extraction generated coherent, interpretable summaries per tweet
