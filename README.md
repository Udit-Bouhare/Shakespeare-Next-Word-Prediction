# 📖 Shakespeare Next-Word Prediction using LSTM & GRU

A Deep Learning project that predicts the **next word in a sequence** using Shakespeare's *Hamlet* as the training corpus. The project implements **LSTM and GRU models using PyTorch** and provides an interactive Streamlit application for text generation.

## 🚀 Live Demo

🔗 **Streamlit App:** https://shakespeare-next-word-predictiongit-lqykcwgknmkdgdypvz8gzl.streamlit.app/

---

## 📌 Project Overview

The goal of this project is to build a language model capable of learning patterns from Shakespeare's *Hamlet* and predicting the next word given a text prompt.

For example:

```text
Input:
If you do meet Horatio

Output:
If you do meet Horatio ...
```

The model generates text **autoregressively**, where each predicted word is added to the input to generate the next word.

---

## ⚙️ NLP Pipeline

```text
Raw Text
   ↓
Tokenization
   ↓
Vocabulary Creation
   ↓
Word → Index
   ↓
Sequence Generation
   ↓
Padding
   ↓
Training Data
   ↓
Embedding
   ↓
LSTM / GRU
   ↓
Next-Word Prediction
```

### Preprocessing

* Shakespeare's *Hamlet* from the NLTK Gutenberg Corpus
* Text normalization
* NLTK tokenization
* Vocabulary creation
* Numerical encoding
* Sequence generation
* Padding

---

## 🧠 Model Architecture

### LSTM

```text
Input
  ↓
Embedding (100)
  ↓
LSTM (150)
  ↓
Linear
  ↓
Vocabulary Scores
```

### GRU

```text
Input
  ↓
Embedding (100)
  ↓
GRU (150)
  ↓
Dropout (0.2)
  ↓
GRU (100)
  ↓
Linear
  ↓
Vocabulary Scores
```

---

## 🏋️ Training

| Parameter      | Value            |
| -------------- | ---------------- |
| Framework      | PyTorch          |
| Optimizer      | Adam             |
| Learning Rate  | 0.001            |
| Loss           | CrossEntropyLoss |
| Batch Size     | 32               |
| Epochs         | 50               |
| Embedding Size | 100              |

---

## 🔮 Text Generation

The trained model takes a text prompt, predicts the most likely next token, appends it to the prompt, and repeats the process to generate multiple words.

```text
Prompt
  ↓
Predict Next Word
  ↓
Append Word
  ↓
Predict Again
  ↓
Generated Text
```

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **LSTM & GRU**
* **NLTK**
* **NumPy**
* **Pandas**
* **Scikit-Learn**
* **Streamlit**

---

## 📁 Project Structure

```text
Shakespeare-Next-Word-Prediction/
│
├── app.py
├── model.py
├── lstm_checkpoint.pth
├── halmet.txt
├── experiments.ipynb
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

---

## 📈 Future Improvements

* Compare LSTM and GRU performance
* Add validation and test datasets
* Improve text preprocessing
* Hyperparameter tuning
* Temperature and Top-K sampling
* Improve text generation quality
* Enhance Streamlit interface

---

## 👨‍💻 Author

**Udit Sharma**

🔗 [LinkedIn](https://www.linkedin.com/in/udit-sharma-5533082aa/)

---

⭐ If you found this project useful, consider giving the repository a star!
