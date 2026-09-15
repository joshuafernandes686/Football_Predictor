import React, { useState, useRef, useEffect } from "react";

export default function TeamSelector({ label, selectedTeam, onSelectTeam, teams, disabled }) {
  const [query, setQuery] = useState(selectedTeam || "");
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    setQuery(selectedTeam || "");
  }, [selectedTeam]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredTeams = query.trim() === ""
    ? teams
    : teams.filter((team) =>
        team.toLowerCase().includes(query.trim().toLowerCase())
      );

  const handleInputChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    setIsOpen(true);
    if (teams.includes(val)) {
      onSelectTeam(val);
    } else {
      onSelectTeam("");
    }
  };

  const handleSelectOption = (team) => {
    setQuery(team);
    onSelectTeam(team);
    setIsOpen(false);
  };

  return (
    <div className="team-selector" ref={containerRef}>
      <label className="selector-label">{label}</label>
      <div className="input-wrapper">
        <input
          type="text"
          className="search-input"
          placeholder="🔍 Search team..."
          value={query}
          onFocus={() => setIsOpen(true)}
          onChange={handleInputChange}
          disabled={disabled}
        />
        {query && (
          <button
            type="button"
            className="clear-btn"
            onClick={() => {
              setQuery("");
              onSelectTeam("");
              setIsOpen(true);
            }}
          >
            ×
          </button>
        )}
      </div>

      {isOpen && (
        <ul className="dropdown-list">
          {filteredTeams.length > 0 ? (
            filteredTeams.slice(0, 50).map((team) => (
              <li
                key={team}
                className={`dropdown-item ${selectedTeam === team ? "active" : ""}`}
                onClick={() => handleSelectOption(team)}
              >
                {team}
              </li>
            ))
          ) : (
            <li className="dropdown-no-results">No team found</li>
          )}
        </ul>
      )}
    </div>
  );
}
