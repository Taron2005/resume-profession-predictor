# Resume Profession Predictor

This project predicts a candidate's profession based on the textual content of their resume. It uses machine learning techniques to classify the text into likely job roles such as Data Scientist, Software Engineer, Accountant, etc.
- Input: Raw text from a resume
- Output: Predicted profession or job title
- Supports text input via a web interface 
- Vectorization using:
  -  TF-IDF (best performance)
  -  BERT (tested, but less effective after deployment)
- Classification using **Logistic Regression**

# How to run locally

Make sure Python is installed, then install the required packages and run:

```bash
pip install -r requirements.txt
streamlit run App.py
