from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
from datetime import datetime
import json
import os

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
def init_db():
    conn = sqlite3.connect('creditpulse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scores
                 (id INTEGER PRIMARY KEY, ticker TEXT, score REAL, 
                  risks TEXT, model TEXT, timestamp TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS metrics
                 (id INTEGER PRIMARY KEY, model TEXT, accuracy REAL,
                  precision_val REAL, recall_val REAL, f1 REAL, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# VADER for sentiment
vader = SentimentIntensityAnalyzer()

# Mock news (replace with NewsAPI if you have key)
MOCK_NEWS = {
    "AAPL": ["Apple announces record revenue", "New iPhone sales exceed expectations", "Strong services growth"],
    "TSLA": ["Tesla faces production delays", "Regulatory scrutiny increases", "Musk warns of challenges"],
    "MSFT": ["Microsoft cloud growth accelerates", "Strong enterprise demand", "AI investments paying off"],
    "GOOGL": ["Google faces antitrust lawsuit", "Ad revenue under pressure", "Search dominance questioned"],
    "AMZN": ["Amazon reports strong Q3", "AWS revenue beats expectations", "Retail margins improve"],
}

# Trained models (global to avoid retraining)
MODELS_TRAINED = False
xgb_model = None
lstm_model = None
dt_model = None

class CreditRequest(BaseModel):
    ticker: str
    model_type: str = "xgboost"  # xgboost, lstm, decision_tree

def get_financial_data(ticker):
    """Fetch Yahoo Finance data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="3mo")
        
        return {
            'debt_to_equity': info.get('debtToEquity', 50) / 100,
            'current_ratio': info.get('currentRatio', 1.5),
            'profit_margin': info.get('profitMargins', 0.1) * 100,
            'revenue_growth': info.get('revenueGrowth', 0.05) * 100,
            'volatility': hist['Close'].pct_change().std() * 100 if len(hist) > 1 else 2.0,
            'market_cap': info.get('marketCap', 1e9) / 1e9,
            'pe_ratio': info.get('trailingPE', 15),
            'roa': info.get('returnOnAssets', 0.05) * 100 if info.get('returnOnAssets') else 5.0
        }
    except:
        return {
            'debt_to_equity': 0.5, 'current_ratio': 1.5, 'profit_margin': 10.0,
            'revenue_growth': 5.0, 'volatility': 2.0, 'market_cap': 100.0,
            'pe_ratio': 15.0, 'roa': 5.0
        }

def analyze_news(ticker):
    """Mock news analysis with VADER (use NewsAPI in production)"""
    news = MOCK_NEWS.get(ticker, ["Company maintains stable operations", "Business as usual"])
    sentiments = [vader.polarity_scores(n)['compound'] for n in news]
    return np.mean(sentiments)

def calculate_risks(ticker, financials, sentiment):
    """Map data to 5 risk categories"""
    risks = {}
    
    # Financial Risk
    financial_score = 100
    if financials['debt_to_equity'] > 1.0:
        financial_score -= 30
    if financials['profit_margin'] < 5:
        financial_score -= 25
    if financials['roa'] < 3:
        financial_score -= 15
    risks['Financial Risk'] = max(0, financial_score)
    
    # Operational Risk
    operational_score = 100
    if financials['current_ratio'] < 1.0:
        operational_score -= 30
    if financials['revenue_growth'] < 0:
        operational_score -= 35
    risks['Operational Risk'] = max(0, operational_score)
    
    # Management/Leadership Risk
    leadership_score = 50 + (sentiment * 50)
    risks['Management Risk'] = max(0, min(100, leadership_score))
    
    # Macroeconomic Risk
    macro_score = 100 - (financials['volatility'] * 10)
    if financials['pe_ratio'] > 30:
        macro_score -= 15
    risks['Macroeconomic Risk'] = max(0, min(100, macro_score))
    
    # Market Sentiment Risk
    market_score = (50 + sentiment * 30) - (financials['volatility'] * 5)
    risks['Market Sentiment Risk'] = max(0, min(100, market_score))
    
    return risks

def prepare_features(financials, risks):
    """Prepare feature vector for ML models"""
    return np.array([[
        financials['debt_to_equity'],
        financials['current_ratio'],
        financials['profit_margin'] / 100,
        financials['revenue_growth'] / 100,
        financials['volatility'] / 100,
        financials['pe_ratio'] / 100,
        financials['roa'] / 100,
        np.mean(list(risks.values())) / 100
    ]])

def train_models():
    """Train all 3 models with synthetic data"""
    global MODELS_TRAINED, xgb_model, lstm_model, dt_model
    
    if MODELS_TRAINED:
        return
    
    # Generate synthetic training data (200 samples)
    np.random.seed(42)
    n_samples = 200
    X_train = np.random.rand(n_samples, 8)
    
    # Create realistic labels (companies with better metrics = lower default risk)
    feature_weights = np.array([0.3, 0.15, 0.2, 0.15, 0.1, 0.05, 0.025, 0.025])
    scores = X_train @ feature_weights
    y_train = (scores > 0.35).astype(int)  # 1 = creditworthy, 0 = risky
    
    # 1. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 2. XGBoost
    xgb_model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    # 3. LSTM (simple sequential model)
    lstm_model = keras.Sequential([
        keras.layers.Input(shape=(8,)),
        keras.layers.Reshape((8, 1)),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Calculate and store metrics
    for model, name in [(dt_model, 'decision_tree'), (xgb_model, 'xgboost'), (lstm_model, 'lstm')]:
        if name == 'lstm':
            y_pred_proba = model.predict(X_train, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_train)
        
        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        f1 = f1_score(y_train, y_pred, zero_division=0)
        
        # Store in DB
        conn = sqlite3.connect('creditpulse.db')
        c = conn.cursor()
        c.execute("INSERT INTO metrics VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                  (name, acc, prec, rec, f1, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    MODELS_TRAINED = True

def predict_with_model(X_input, model_type):
    """Predict using specified model"""
    train_models()
    
    if model_type == "xgboost":
        prob = xgb_model.predict_proba(X_input)[0][1]
        model = xgb_model
    elif model_type == "lstm":
        prob = lstm_model.predict(X_input, verbose=0)[0][0]
        model = None  # SHAP doesn't work well with LSTM
    else:  # decision_tree
        prob = dt_model.predict_proba(X_input)[0][1]
        model = dt_model
    
    score = prob * 100
    
    # SHAP explanations (only for tree-based)
    explanations = []
    if model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            
            feature_names = ['Debt/Equity', 'Current Ratio', 'Profit Margin', 
                           'Revenue Growth', 'Volatility', 'P/E Ratio', 'ROA', 'Risk Score']
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]
            
            explanations = [
                {'feature': name, 'impact': float(val)}
                for name, val in zip(feature_names, shap_vals)
            ]
        except:
            explanations = [{'feature': 'SHAP unavailable for this model', 'impact': 0.0}]
    else:
        explanations = [{'feature': 'SHAP not supported for LSTM', 'impact': 0.0}]
    
    return score, explanations

@app.get("/")
def read_root():
    return {"message": "CreditPulse API v2.0 - XGBoost + LSTM + Metrics"}

@app.post("/analyze")
def analyze_credit(request: CreditRequest):
    ticker = request.ticker.upper()
    model_type = request.model_type
    
    try:
        # 1. Fetch financial data
        financials = get_financial_data(ticker)
        
        # 2. Analyze news sentiment
        sentiment = analyze_news(ticker)
        
        # 3. Calculate risk categories
        risks = calculate_risks(ticker, financials, sentiment)
        
        # 4. Prepare features
        X_input = prepare_features(financials, risks)
        
        # 5. ML scoring + SHAP
        score, explanations = predict_with_model(X_input, model_type)
        
        # 6. Save to DB
        conn = sqlite3.connect('creditpulse.db')
        c = conn.cursor()
        c.execute("INSERT INTO scores VALUES (NULL, ?, ?, ?, ?, ?)",
                  (ticker, score, json.dumps(risks), model_type, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        return {
            "ticker": ticker,
            "score": round(score, 2),
            "model": model_type,
            "risks": risks,
            "explanations": explanations,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{ticker}")
def get_history(ticker: str):
    conn = sqlite3.connect('creditpulse.db')
    c = conn.cursor()
    c.execute("SELECT score, risks, model, timestamp FROM scores WHERE ticker=? ORDER BY timestamp DESC LIMIT 10", 
              (ticker.upper(),))
    rows = c.fetchall()
    conn.close()
    
    history = [
        {
            "score": row[0],
            "risks": json.loads(row[1]),
            "model": row[2],
            "timestamp": row[3]
        }
        for row in rows
    ]
    
    return {"history": history}

@app.get("/metrics")
def get_metrics():
    """Get model performance metrics"""
    train_models()  # Ensure models are trained
    
    conn = sqlite3.connect('creditpulse.db')
    c = conn.cursor()
    c.execute("SELECT model, accuracy, precision_val, recall_val, f1 FROM metrics ORDER BY timestamp DESC LIMIT 3")
    rows = c.fetchall()
    conn.close()
    
    metrics = [
        {
            "model": row[0],
            "accuracy": round(row[1] * 100, 2),
            "precision": round(row[2] * 100, 2),
            "recall": round(row[3] * 100, 2),
            "f1_score": round(row[4] * 100, 2)
        }
        for row in rows
    ]
    
    return {"metrics": metrics}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)