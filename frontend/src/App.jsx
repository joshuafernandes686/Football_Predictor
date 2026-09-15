import React, { useState, useEffect } from "react";
import TeamSelector from "./components/TeamSelector";
import ShapChart from "./components/ShapChart";
import FootballChatbot from "./components/FootballChatbot";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

export default function App() {
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [loadingTeams, setLoadingTeams] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [validationError, setValidationError] = useState("");
  const [apiError, setApiError] = useState("");
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    async function fetchTeams() {
      try {
        setLoadingTeams(true);
        setApiError("");
        const res = await fetch(`${API_BASE_URL}/teams`);
        if (!res.ok) {
          throw new Error(`Failed to load teams (HTTP ${res.status})`);
        }
        const data = await res.json();
        setTeams(data);
      } catch (err) {
        console.error("Error fetching teams:", err);
        setApiError("Backend unavailable. Please ensure python main.py is running on http://127.0.0.1:8000.");
      } finally {
        setLoadingTeams(false);
      }
    }
    fetchTeams();
  }, []);

  const handlePredict = async () => {
    setValidationError("");
    setApiError("");

    if (!homeTeam) {
      setValidationError("Please select a Home Team.");
      return;
    }
    if (!awayTeam) {
      setValidationError("Please select an Away Team.");
      return;
    }
    if (homeTeam.toLowerCase() === awayTeam.toLowerCase()) {
      setValidationError("Home Team and Away Team must be different!");
      return;
    }

    try {
      setPredicting(true);
      setPredictionResult(null);

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          home_team: homeTeam,
          away_team: awayTeam,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Prediction failed (HTTP ${res.status})`);
      }

      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setApiError(err.message || "Failed to generate prediction. Check backend status.");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">⚽ FOOTBALL PREDICTOR</h1>
        <p className="app-subtitle">Predict match outcomes with Machine Learning & RAG Q&A</p>
      </header>

      <main className="main-content">
        {/* Prediction Form Card */}
        <div className="card form-card">
          {loadingTeams ? (
            <div className="loading-spinner">Loading dataset teams...</div>
          ) : (
            <>
              <div className="selectors-grid">
                <TeamSelector
                  label="HOME TEAM"
                  selectedTeam={homeTeam}
                  onSelectTeam={(team) => {
                    setHomeTeam(team);
                    setValidationError("");
                  }}
                  teams={teams}
                  disabled={predicting}
                />

                <div className="vs-badge">VS</div>

                <TeamSelector
                  label="AWAY TEAM"
                  selectedTeam={awayTeam}
                  onSelectTeam={(team) => {
                    setAwayTeam(team);
                    setValidationError("");
                  }}
                  teams={teams}
                  disabled={predicting}
                />
              </div>

              {validationError && <div className="alert alert-warning">{validationError}</div>}
              {apiError && <div className="alert alert-error">{apiError}</div>}

              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={predicting || !homeTeam || !awayTeam}
              >
                {predicting ? "Analyzing match..." : "[ PREDICT ]"}
              </button>
            </>
          )}
        </div>

        {/* Prediction Output Results */}
        {predictionResult && (
          <div className="results-wrapper">
            {/* Match Header & Probabilities */}
            <div className="card result-card">
              <h2 className="match-title">
                {predictionResult.home_team} <span className="vs-light">vs</span> {predictionResult.away_team}
              </h2>

              <div className="probabilities-container">
                <div className="prob-row">
                  <span className="prob-label">{predictionResult.home_team} Win</span>
                  <div className="prob-bar-wrapper">
                    <div
                      className="prob-bar home-bar"
                      style={{ width: `${(predictionResult.probabilities?.home || 0) * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">
                    {Math.round((predictionResult.probabilities?.home || 0) * 100)}%
                  </span>
                </div>

                <div className="prob-row">
                  <span className="prob-label">Draw</span>
                  <div className="prob-bar-wrapper">
                    <div
                      className="prob-bar draw-bar"
                      style={{ width: `${(predictionResult.probabilities?.draw || 0) * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">
                    {Math.round((predictionResult.probabilities?.draw || 0) * 100)}%
                  </span>
                </div>

                <div className="prob-row">
                  <span className="prob-label">{predictionResult.away_team} Win</span>
                  <div className="prob-bar-wrapper">
                    <div
                      className="prob-bar away-bar"
                      style={{ width: `${(predictionResult.probabilities?.away || 0) * 100}%` }}
                    />
                  </div>
                  <span className="prob-value">
                    {Math.round((predictionResult.probabilities?.away || 0) * 100)}%
                  </span>
                </div>
              </div>

              {/* Winner Announcement Card */}
              <div className="winner-box">
                <span className="winner-subtitle">PREDICTED RESULT</span>
                <h3 className="winner-name">{predictionResult.prediction.toUpperCase()}</h3>
              </div>
            </div>

            {/* Reasons Card */}
            {predictionResult.reasons && predictionResult.reasons.length > 0 && (
              <div className="card reasons-card">
                <h3 className="section-title">💡 WHY THIS PREDICTION?</h3>
                <ul className="reasons-list">
                  {predictionResult.reasons.map((reason, idx) => (
                    <li key={idx} className="reason-item">
                      • {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* SHAP Chart Card */}
            {predictionResult.shap && (
              <div className="card shap-card">
                <ShapChart shapData={predictionResult.shap} />
              </div>
            )}

            {/* Match-Aware Contextual RAG Chatbot Card */}
            <div className="card match-chat-card">
              <h3 className="section-title">🤖 Ask about this prediction</h3>
              <FootballChatbot
                defaultHomeTeam={predictionResult.home_team}
                defaultAwayTeam={predictionResult.away_team}
                predictionContext={predictionResult}
              />
            </div>
          </div>
        )}

        {/* General Football RAG Chatbot Section */}
        {!predictionResult && (
          <div className="general-chatbot-section">
            <FootballChatbot />
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Football Predictor System • ML Classifier & Local RAG Architecture</p>
      </footer>
    </div>
  );
}
