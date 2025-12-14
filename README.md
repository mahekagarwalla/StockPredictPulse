# StockPredictPulse 🚀

A **FastAPI-based machine learning API** for stock market prediction and financial sentiment analysis. This project integrates economic indicators, market data, and sentiment signals to provide predictive insights, packaged with Docker for easy deployment.

---

## ✨ Features

* FastAPI backend for high-performance APIs
* Machine Learning models using **TensorFlow**, **XGBoost**, and **scikit-learn**
* Financial data from **Yahoo Finance** and **FRED API**
* News sentiment analysis using **VADER**
* SHAP-based model explainability
* Load testing support
* Dockerized for reproducible deployment



## ⚙️ Tech Stack

* **Backend**: FastAPI, Uvicorn
* **ML / Data**: TensorFlow, XGBoost, scikit-learn, pandas, numpy
* **Finance APIs**: yfinance, fredapi, newsapi
* **Explainability**: SHAP
* **DevOps**: Docker

---

## 🚀 Getting Started (Local Setup)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mahekagarwalla/StockPredictPulse.git
cd StockPredictPulse
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\\Scripts\\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the API

```bash
uvicorn app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Swagger docs:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker Usage

### Build the Docker image

```bash
docker build -t stockpredictpulse .
```

### Run the container

```bash
docker run -p 8000:8000 stockpredictpulse
```

---

## 🧪 Load Testing

To test API performance:

```bash
python load_test.py
```

---

## 🎯 Why This Project Matters (For Recruiters)

* Demonstrates **end-to-end ML system design**
* Shows real-world **financial data integration**
* Uses **production-grade backend (FastAPI)**
* Dockerized → deployment-ready
* Includes explainability (SHAP) — industry-relevant

---

## 📌 Future Improvements

* Add CI/CD pipeline
* Model versioning
* Cloud deployment (AWS/GCP/Azure)
* Authentication & rate limiting

---

## 👩‍💻 Author

**Mahek Agarwalla**
B.Tech CSE | Machine Learning & Full-Stack Development

---

⭐ If you like this project, consider giving it a star!
