import React from "react";

export default function ShapChart({ shapData }) {
  if (!shapData || shapData.length === 0) return null;

  // Calculate maximum absolute impact for percentage scaling
  const maxImpact = Math.max(...shapData.map((item) => Math.abs(item.impact)), 0.001);

  return (
    <div className="shap-container">
      <h3 className="section-title">📊 SHAP EXPLANATION</h3>
      <p className="section-subtitle">
        Local feature impact on model prediction (higher bar = greater influence)
      </p>

      <div className="shap-list">
        {shapData.map((item, idx) => {
          const barWidth = Math.min(100, Math.max(8, (Math.abs(item.impact) / maxImpact) * 100));
          const isPositive = item.direction === "positive" || item.impact >= 0;

          return (
            <div key={idx} className="shap-item">
              <div className="shap-header">
                <span className="shap-feature-name">{item.feature}</span>
                <span className="shap-feature-value">
                  Val: {item.value} | Impact: {item.impact > 0 ? `+${item.impact}` : item.impact}
                </span>
              </div>
              <div className="shap-bar-track">
                <div
                  className={`shap-bar-fill ${isPositive ? "positive" : "negative"}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
