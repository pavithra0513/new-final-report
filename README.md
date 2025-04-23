#  NLP-Based Resume Parsing Using Named Entity Recognition (NER)
## 1. Introduction
Named Entity Recognition (NER) is a subfield of Natural Language Processing (NLP) focused on identifying and categorizing entities in unstructured text (e.g., names, dates, organizations, and skills). 

This project addresses the inefficiency of traditional rule-based resume parsers when faced with inconsistent formats or ambiguous terms. We explore multiple machine learning and deep learning models including:

- **CRF (Conditional Random Fields)**
- **BiLSTM-CRF**
- **Transformers (BERT, RoBERTa)**

The models are benchmarked using data science principles like supervised learning, sequence labeling, and contextual embeddings.
## 2. Methods

This section describes the methodologies used, with attention to clarity and replicability.

### 2.1 Rule-Based Approach

- Extracted entities like phone numbers and emails using regular expressions.
- Fast and deterministic but fails with non-standard formats.

### 2.2 Conditional Random Fields (CRF)

CRFs are used for sequence labeling by modeling the conditional probability of label sequences.
**CRF Equation:**
P(y|x) = (1 / Z(x)) * exp(Σ_t=1^T Σ_k λ_k f_k(y_{t-1}, y_t, x, t))
- Best for structured fields
- Relies heavily on engineered features
### 2.3 BiLSTM-CRF

Combines the strength of BiLSTM (context capture) and CRF (structured prediction).

- LSTM Hidden State: `h_t = LSTM(x_t, h_{t-1})`
- BiLSTM Output: `h_t = [→h_t; ←h_t]`
### 2.4 Transformer-Based Models (BERT, RoBERTa)

Fine-tuned BERT and RoBERTa for token classification tasks using pre-labeled resume data.

**Self-Attention Equation:**
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
- Best performance in contextual ambiguity
- Higher computational cost
### 2.5 Training Configuration

- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 32  
- **Learning Rate:** 0.0001  
- **Epochs:** 10–20  
- **Evaluation Metrics:** Precision, Recall, F1 Score, Confusion Matrix, ROC
 ## 3. Results

### 3.1 Performance Overview

| Model       | Accuracy | F1 Score | Notes                             |
|-------------|----------|----------|-----------------------------------|
| CRF         | 90%      | 0.89     | Structured, requires features     |
| spaCy NER   | 93%      | 0.91     | Lightweight, good generalization  |
| BERT        | 95%      | 0.93     | Best contextual understanding     |

### 3.2 Observations

- All models effectively extract `Name`, `Email`, and `Skills`.
- Struggle with fields like `Graduation Year` and `Experience Duration`.
- BiLSTM-CRF showed signs of overfitting in small datasets.

### 3.3 Planned Visualizations

- Training & validation loss/accuracy curves
- Entity-wise F1 Score comparison (bar chart)
- ROC Curves per entity type
## 4. Discussion

- **Transformers (BERT/RoBERTa)** show superior performance due to context modeling
- **Rule-based systems** are fast but brittle under format variability
- **BiLSTM-CRF** balances structure and sequence, but is compute-heavy
- **Labeling inconsistencies** in datasets can impact model learning
- Ambiguous entities like `"Python"` are best resolved via deep contextual embeddings
## 5. Future Work

- Extend system to **multilingual resumes** (using `mBERT`, `XLM-R`)
- Add **domain-specific ontologies** (healthcare, finance)
- Use **active learning** to reduce labeling effort
- Apply **bias detection/fairness-aware models**
- Deploy as **API-based real-time parser** for integration into hiring systems
## References

-  [Resume NER Dataset (GitHub)](https://github.com/vrundag91/Resume-Corpus-Dataset)  
-  [spaCy NLP Library](https://spacy.io/)  
-  Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*  
-  Lafferty et al. (2001), *Conditional Random Fields for Labeling Sequential Data*  
-  Tjong Kim Sang & De Meulder (2003), *CoNLL-2003 Shared Task*  
-  Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese Networks*
