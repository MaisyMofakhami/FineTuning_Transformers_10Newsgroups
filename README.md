# 🧠 Fine-Tuning Transformers for Multi-Class Text Classification

This project explores transformer-based models (e.g., BERT, RoBERTa) for multi-class text classification on the **20 Newsgroups** dataset. We delve into preprocessing, fine-tuning, model interpretability, and scalability of transformer architectures for real-world NLP tasks.

---

## 📚 Objective

To deepen understanding of transformer architectures by:

* Fine-tuning pre-trained models on the 20 Newsgroups dataset
* Experimenting with regularization and classifier heads
* Analyzing interpretability using attention and SHAP/LIME
* Exploring model deployment options

---

## 📁 Dataset

**20 Newsgroups** — \~20,000 documents from 20 distinct topics.

* Loaded using `sklearn.datasets.fetch_20newsgroups`
* Stratified train/validation/test split

---

## 𞽹 Data Preprocessing

* Removed stopwords, applied stemming
* Handled OOV words using tokenizer vocabularies
* Tokenized using HuggingFace's tokenizer
* Managed class imbalance via stratified sampling

---

## 🧪 Data Augmentation

To improve model generalizability and reduce overfitting, we applied the following data augmentation strategies:

* **Synonym Replacement**: Randomly substituted non-stopwords with their synonyms using WordNet, preserving semantic meaning while introducing lexical diversity.
* **Random Deletion**: Randomly removed low-importance tokens from training samples to simulate noisy input and enhance model robustness.

These techniques helped simulate lexical variability and improved the model's ability to generalize to unseen data.

---

## 🤖 Models & Training

* Pre-trained models: **BERT**, **RoBERTa**
* Added custom classification heads:

  * Dense layers
* Regularization:

  * Tuned dropout rates and learning rates
  * Used `AdamW` optimizer and schedulers
* Mixed-precision training was leveraged where available
* Early stopping and checkpointing based on validation performance

---

## 📊 Evaluation

* Metrics: Precision, Recall, F1-score, and per-class breakdown

![image](https://github.com/user-attachments/assets/81677698-320b-454f-8f83-0def416831d5)

* Attention visualizations to interpret focus regions

### 📈 Example: Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### 🛝 Attention Map Example

![Attention Visualization](images/attention_weights.png)

---

## 🧠 Interpretability

Used **LIME** and **SHAP** for model interpretability:

* Visualized token-wise contribution to predictions
* Compared attention with feature attribution maps

### Example: SHAP Explanation

![image](https://github.com/user-attachments/assets/c33be094-f557-48eb-b0fe-bf0052d27f5d)

---

## 📌 Future Work

* Combine multiple datasets for transfer learning
* Add multilingual transformers (XLM-R, mBERT)
* Explore low-rank fine-tuning (LoRA, AdapterFusion)
* Deploy the model using ONNX or TorchServe
