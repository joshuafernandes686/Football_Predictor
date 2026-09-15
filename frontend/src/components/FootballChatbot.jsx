import React, { useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

export default function FootballChatbot({ defaultHomeTeam, defaultAwayTeam, predictionContext }) {
  const [messages, setMessages] = useState([
    {
      sender: "assistant",
      text: "Hello! I am FootballPredictor AI. Ask me any question about international football teams, historical match results, head-to-head records, or team stats.",
      sources: [],
    },
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const suggestedQuestions = [
    defaultHomeTeam && defaultAwayTeam
      ? `What is ${defaultHomeTeam}'s historical record against ${defaultAwayTeam}?`
      : "How has France performed recently?",
    defaultHomeTeam ? `Tell me about ${defaultHomeTeam}'s recent results.` : "What is Spain's H2H record vs France?",
    "Tell me about Germany's recent results.",
  ];

  const handleSend = async (questionText) => {
    const textToSend = questionText || inputQuery;
    if (!textToSend.trim() || loading) return;

    setErrorMsg("");
    const userMsg = { sender: "user", text: textToSend };
    setMessages((prev) => [...prev, userMsg]);
    if (!questionText) setInputQuery("");

    try {
      setLoading(true);
      const res = await fetch(`${API_BASE_URL}/rag/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: textToSend,
          home_team: defaultHomeTeam || null,
          away_team: defaultAwayTeam || null,
          prediction_context: predictionContext || null,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Chat request failed (HTTP ${res.status})`);
      }

      const data = await res.json();
      const assistantMsg = {
        sender: "assistant",
        text: data.answer || "No response received.",
        sources: data.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error("RAG Chat error:", err);
      setErrorMsg(err.message || "Failed to query RAG engine.");
      setMessages((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "Sorry, I encountered an error searching the football knowledge base. Please check backend connection.",
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([
      {
        sender: "assistant",
        text: "Conversation cleared. Ask me any question about international football data!",
        sources: [],
      },
    ]);
    setErrorMsg("");
  };

  return (
    <div className="chatbot-card">
      <div className="chatbot-header">
        <div>
          <h3 className="chatbot-title">⚽ Ask FootballPredictor AI</h3>
          <p className="chatbot-subtitle">
            Retrieval-Augmented Q&A grounded in historical FIFA match data
          </p>
        </div>
        <button type="button" className="clear-chat-btn" onClick={handleClear} title="Clear Conversation">
          Clear
        </button>
      </div>

      {/* Suggested Question Chips */}
      <div className="chips-container">
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            type="button"
            className="chip-btn"
            onClick={() => handleSend(q)}
            disabled={loading}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Chat Thread */}
      <div className="chat-thread">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble-wrapper ${msg.sender}`}>
            <div className={`chat-bubble ${msg.sender}`}>
              <div className="bubble-text">{msg.text}</div>

              {/* Source Attribution Badges */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources-container">
                  <span className="sources-header">📚 Grounded Sources:</span>
                  <div className="sources-list">
                    {msg.sources.map((src, sIdx) => (
                      <div key={sIdx} className="source-tag">
                        <span className="source-match">
                          {src.team} vs {src.opponent}
                        </span>
                        <span className="source-meta">
                          {src.tournament} ({src.date})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-bubble-wrapper assistant">
            <div className="chat-bubble assistant loading-bubble">
              🔍 Searching football knowledge base...
            </div>
          </div>
        )}
      </div>

      {errorMsg && <div className="alert alert-error">{errorMsg}</div>}

      {/* Input Box */}
      <div className="chat-input-wrapper">
        <input
          type="text"
          className="chat-input"
          placeholder="Ask anything about football data..."
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          type="button"
          className="chat-send-btn"
          onClick={() => handleSend()}
          disabled={loading || !inputQuery.trim()}
        >
          {loading ? "..." : "Ask"}
        </button>
      </div>
    </div>
  );
}
