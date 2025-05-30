# 🧠 NLP - Detection of the Problem Source from Arabic Feedback

This project uses Natural Language Processing (NLP) to classify Arabic customer feedback as either related to the **product** (المنتج) or the **service** (الخدمة). It includes a trained model, a Streamlit web app, and a Jupyter Notebook for training and evaluation.

---

## 📂 Project Structure
```plaintext
tunisian-feedback-classifier/
│
├── app.py # Streamlit app interface
├── NLP_Classification.ipynb # Notebook with full training & evaluation
├── MLP_model.pkl # Trained MLP neural network model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── finals.csv # Contains the data (Comment_Text_Arabic,Problem_Source(Labels))
├── requirements.txt #Contains the requirements to make the interface work
├── README.md # Project documentation (this file)
```
---

## 🚀 Features

- Classifies Arabic feedback as either about the **product** or the **service**
- Neural network (MLP) trained on TF-IDF features
- Arabic-specific text preprocessing (normalization + stopword removal)
- Interactive web app with Streamlit

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YoussefNKH/tunisian-feedback-classifier.git
cd tunisian-feedback-classifier
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---
## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---
## 📈 Model Performance

| Model         | Accuracy | F1-Score |
|---------------|----------|----------|
| Naive Bayes   | 0.92     | 0.92     |
| Neural Net ✅ | 0.95     | 0.95     |
| RBF SVM       | 0.95     | 0.95     |
| Linear SVM    | 0.93     | 0.93     |

---
## 👤 Author

**Youssef Nakhli**  
🎓 Data Engineer Student   
📫 [LinkedIn](https://www.linkedin.com/in/youssef-nakhli-804946277/) | [GitHub](https://github.com/YoussefNKH)
