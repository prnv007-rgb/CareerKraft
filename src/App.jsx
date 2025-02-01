import { useState } from "react";
import "./App.css";

function App() {
  const [resumes, setResumes] = useState([]);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("Pending");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const addResume = () => {
    if (!file) return alert("Please select a resume file!");

    const newResume = {
      id: Date.now(),
      filename: file.name,
      status,
      fileURL: URL.createObjectURL(file),
      analysis: "", // Placeholder for NLP data
      expanded: false, // To toggle analysis details
    };

    setResumes([...resumes, newResume]);
    setFile(null);
  };

  const updateStatus = (id, newStatus) => {
    setResumes(
      resumes.map((resume) =>
        resume.id === id ? { ...resume, status: newStatus } : resume
      )
    );
  };

  const deleteResume = (id) => {
    setResumes(resumes.filter((resume) => resume.id !== id));
  };

  const toggleDetails = (id) => {
    setResumes(
      resumes.map((resume) =>
        resume.id === id ? { ...resume, expanded: !resume.expanded } : resume
      )
    );
  };

  return (
    <div className="container">
      <h1>📄 Resume Tracker</h1>

      <div className="form">
        <input type="file" accept=".pdf,.doc,.docx" onChange={handleFileChange} />
        <button onClick={addResume}>➕ Upload Resume</button>
      </div>

      {resumes.length === 0 ? (
        <p className="empty">No resumes uploaded yet!</p>
      ) : (
        <ul>
          {resumes.map((resume) => (
            <li key={resume.id} className="resume-card">
              <div className="resume-info">
                <a href={resume.fileURL} target="_blank" rel="noopener noreferrer">
                  📂 {resume.filename}
                </a>
                <select
                  value={resume.status}
                  onChange={(e) => updateStatus(resume.id, e.target.value)}
                >
                  <option value="Pending">🟡 Pending</option>
                  <option value="Reviewed">🟢 Reviewed</option>
                  <option value="Rejected">🔴 Rejected</option>
                </select>
              </div>

              <button className="toggle-btn" onClick={() => toggleDetails(resume.id)}>
                {resume.expanded ? "▲ Hide Analysis" : "▼ View Analysis"}
              </button>

              {resume.expanded && (
                <div className="analysis">
                  <p><strong>Analysis Results:</strong></p>
                  <p>(NLP results will be shown here)</p> {/* Placeholder for NLP team */}
                </div>
              )}

              <button className="delete-btn" onClick={() => deleteResume(resume.id)}>
                ❌
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
