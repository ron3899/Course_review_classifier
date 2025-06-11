# 📚 Intelligent Pipeline for Course Review Classification

A modular and comparative approach for classifying course reviews using both classical machine learning (TF-IDF + Naive Bayes) and deep learning (BERT), aimed at automating feedback analysis in educational settings.

---

## 📌 Problem Statement

Manually analyzing large volumes of course reviews is time-consuming, subjective, and inconsistent.  
Our goal: create an **automated, intelligent system** capable of accurately interpreting course reviews and assigning labels like:

- Teaching quality  
- Difficulty  
- Usefulness  
- Interest

We compare a simple “dumber” model (TF-IDF + Naive Bayes) against a more sophisticated “smart” model (BERT), and evaluate both performance and consistency.

---

## 🎯 Project Goals

- **Automation**: Save time and reduce load on human analysts
- **Consistency**: Avoid personal labeling biases
- **Scalability**: Enable large-scale processing and integration with APIs
- **Comparison**: Quantitatively compare classic ML vs. Transformer-based models

---

## 📁 Dataset

We used two main datasets:

1. **Manually-labeled dataset**  
   - 1000 real course reviews from Kaggle  
   - Labeled by 4 different annotators (250 each)  
   - Discarded due to inconsistency

2. **GPT-labeled dataset**  
   - 3500 reviews synthetically generated using GPT-4o  
   - Prompt-engineered for each label and combination  
   - Manually verified samples for quality

**Note**: Label values range from 0–3 for each category.

---

## ⚙️ Methodology

### 🛠 Models

- **Baseline**:  
  `TF-IDF` + `Multinomial Naive Bayes` (wrapped in `MultiOutputClassifier`)

- **Advanced**:  
  `BERT (bert-base-multilingual-cased)` + 4 classification heads

### 🔬 Training Strategy

- Manual dataset used only for analysis (rejected due to inconsistencies)
- GPT-labeled dataset split:
  - 72% Train
  - 8% Validation
  - 20% Test
- Stratified sampling to ensure all label values appear in all splits
- Grid search to optimize hyperparameters:
  - Learning rate
  - Batch size
  - Weight decay
  - Epochs

---

## 📊 Example

**Review**: "Didn't learn anything."

**Target Labels**:
- difficulty = 0  
- interest = 0  
- usefulness = 1  
- teaching_quality = 1  

---

## 🧪 Evaluation Metrics

- Accuracy (for predicting all 4 labels correctly)
- Per-label Precision, Recall, F1-score
- Confusion matrix analysis
- External evaluation by **GPT-4o (zero-shot)**

---

## 🧠 Results Summary

### TF-IDF + Naive Bayes

- **Manual dataset**: ~30% accuracy (4/4 labels) — rejected
- **GPT-labeled dataset**:
  - Worst config: 53%
  - Best config:  
    - 80% (validation)  
    - 77% (evaluation)  
    - >90% per-label accuracy
- Failed mostly on short reviews (<10 words)
- Struggled with scores 1 and 3 (bias toward 0)

### BERT

- Accuracy: **90%**
- Stronger performance on short texts
- Still slight bias:
  - Score 1 often predicted as 0
  - Difficulty score 3 → predicted as 0
- Outperformed baseline significantly

### GPT-4o (Zero-shot Evaluator)

- Accuracy: **55%**
- Heavy bias toward predicting 0
- No clear improvement with longer reviews
- Not suitable as a classifier without fine-tuning

---

## 📝 Key Findings

- **Manual labeling is unreliable** — clear guidelines are essential
- **Short reviews** significantly lower model accuracy
- **TF-IDF model** suffers from high label imbalance
- **BERT** shows robustness and stability across label types
- **GPT-4o**, despite being advanced, is not reliable zero-shot

---

## 🚀 Recommendations

- Build a **larger dataset focused on short reviews**
- Create a **standardized labeling policy** for ambiguous cases
- Fine-tune transformer models for better accuracy and generalization
- Consider hybrid pipelines — e.g., SBERT for detection + GPT for rewriting

---

## 🏁 Final Accuracy Comparison

| Model        | Accuracy (4/4 Labels) |
|--------------|------------------------|
| **BERT**     | ≈ 90% ✅               |
| **Naive Bayes (Baseline)** | ≈ 77%         |
| **GPT-4o (Zero-shot)**     | 55% ❌         |

---

## 🤝 Team

- Gleb Lukovsski  
- Valentin Gundorov  
- Ron Twito  
- Omer Zaadi

---

