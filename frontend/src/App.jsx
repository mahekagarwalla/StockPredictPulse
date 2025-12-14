import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Legend } from 'recharts';

const API_URL = 'http://localhost:8000';

function App() {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const analyzeCredit = async () => {
    if (!ticker) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker.toUpperCase() })
      });
      
      const data = await response.json();
      setResult(data);
      
      // Fetch history
      const histResponse = await fetch(`${API_URL}/history/${ticker}`);
      const histData = await histResponse.json();
      setHistory(histData.history);
      
    } catch (error) {
      alert('Error: ' + error.message);
    }
    setLoading(false);
  };

  const getScoreColor = (score) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRiskColor = (risk) => {
    if (risk >= 70) return 'bg-green-500';
    if (risk >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-indigo-900 mb-4">
            CreditPulse
          </h1>
          <p className="text-xl text-gray-600">
            Real-Time Explainable Credit Intelligence
          </p>
        </div>

        {/* Search */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              placeholder="Enter ticker (e.g., AAPL, TSLA)"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && analyzeCredit()}
              className="flex-1 px-4 py-3 border-2 border-indigo-300 rounded-lg focus:outline-none focus:border-indigo-500 text-lg"
            />
            <button
              onClick={analyzeCredit}
              disabled={loading}
              className="px-8 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 font-semibold text-lg transition"
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-8">
            {/* Credit Score */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Credit Score for {result.ticker}
              </h2>
              <div className="text-center">
                <div className={`text-7xl font-bold ${getScoreColor(result.score)}`}>
                  {result.score}
                </div>
                <div className="text-gray-500 text-lg mt-2">out of 100</div>
                <div className="mt-4 w-full bg-gray-200 rounded-full h-4">
                  <div
                    className={`h-4 rounded-full ${
                      result.score >= 70 ? 'bg-green-500' :
                      result.score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${result.score}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Risk Breakdown */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">
                Risk Category Breakdown
              </h2>
              <div className="space-y-4">
                {Object.entries(result.risks).map(([category, score]) => (
                  <div key={category}>
                    <div className="flex justify-between mb-2">
                      <span className="font-semibold text-gray-700">{category}</span>
                      <span className="font-bold">{score.toFixed(1)}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full ${getRiskColor(score)}`}
                        style={{ width: `${score}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* SHAP Explanations */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">
                Feature Impact (SHAP Values)
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={result.explanations}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="impact" fill="#4F46E5" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-600 mt-4">
                Positive values increase credit score, negative values decrease it
              </p>
            </div>

            {/* Historical Trend */}
            {history.length > 1 && (
              <div className="bg-white rounded-xl shadow-lg p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">
                  Historical Score Trend
                </h2>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={[...history].reverse()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(val) => new Date(val).toLocaleDateString()}
                    />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="score" stroke="#4F46E5" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);