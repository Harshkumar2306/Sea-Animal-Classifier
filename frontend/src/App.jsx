import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, BookOpen, AlertCircle, RefreshCcw, ArrowRight, Play, Trash2, PlusCircle } from 'lucide-react';
import './index.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // History is an array of objects: { id, file, preview, name, predictions, agentResearch, status, error }
  // status: 'idle', 'analyzing', 'done', 'error'
  const [history, setHistory] = useState([]);
  const [activeId, setActiveId] = useState(null);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add('drag-active');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const processFiles = (fileList) => {
    Array.from(fileList).forEach(file => {
      if (file.type.startsWith('image/')) {
        const id = Date.now().toString() + Math.random().toString(36).substring(2);
        const reader = new FileReader();
        
        reader.onload = (e) => {
          const newItem = {
            id,
            file,
            name: file.name,
            preview: e.target.result,
            predictions: [],
            agentResearch: null,
            status: 'idle',
            error: null,
            timestamp: new Date()
          };
          setHistory(prev => [newItem, ...prev]);
          // Automatically shift view to the newly added item
          setActiveId(id);
        };
        
        reader.readAsDataURL(file);
      }
    });
  };

  const updateHistoryItem = (id, updates) => {
    setHistory(prev => prev.map(item => item.id === id ? { ...item, ...updates } : item));
  };

  const analyzeItem = async (id) => {
    const item = history.find(h => h.id === id);
    if (!item || item.status === 'analyzing') return;

    updateHistoryItem(id, { status: 'analyzing', error: null, predictions: [], agentResearch: null });

    const formData = new FormData();
    formData.append('file', item.file);

    try {
      // 1. Predict
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const preds = response.data.predictions;
      updateHistoryItem(id, { predictions: preds });
      
      // 2. Agent Research
      if (preds && preds.length > 0) {
        try {
          const agentRes = await axios.post(`${API_URL}/agent/research`, { label: preds[0].label });
          if (agentRes.data.error) {
            updateHistoryItem(id, { agentResearch: { summary: `Data unavailable: ${agentRes.data.error}` } });
          } else {
            updateHistoryItem(id, { agentResearch: agentRes.data });
          }
        } catch (err) {
          updateHistoryItem(id, { agentResearch: { summary: 'Failed to retrieve research data.' } });
        }
      }
      
      updateHistoryItem(id, { status: 'done' });
    } catch (err) {
      updateHistoryItem(id, { 
        status: 'error', 
        error: err.response?.data?.detail || err.message || 'Error analyzing image.' 
      });
    }
  };

  const analyzeAllPending = () => {
    const pendingItems = history.filter(h => h.status === 'idle');
    pendingItems.forEach(item => analyzeItem(item.id));
  };

  const removeItem = (id, e) => {
    e.stopPropagation();
    setHistory(prev => {
      const newHistory = prev.filter(h => h.id !== id);
      if (activeId === id) {
        setActiveId(newHistory.length > 0 ? newHistory[0].id : null);
      }
      return newHistory;
    });
  };

  const activeItem = history.find(h => h.id === activeId);
  const hasPending = history.some(h => h.status === 'idle');

  return (
    <div className="app-container">
      <header className="header">
        <div>
          <h1>Sea Animal Classifier</h1>
          <p>Identify marine species with neural networks.</p>
        </div>
        {history.length > 0 && (
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={() => fileInputRef.current?.click()}>
              <PlusCircle size={16} /> Add More
            </button>
            <input 
              type="file" 
              multiple 
              ref={fileInputRef} 
              onChange={(e) => processFiles(e.target.files)} 
              accept="image/*" 
              style={{ display: 'none' }} 
            />
          </div>
        )}
      </header>

      {history.length === 0 ? (
        <main className="main-content">
          <div className="upload-center">
            <div 
              className="upload-zone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="upload-icon" />
              <div className="upload-title">Select Images</div>
              <div className="upload-subtitle">Drag and drop multiple images or click to browse</div>
              <input 
                type="file" 
                multiple
                ref={fileInputRef} 
                onChange={(e) => processFiles(e.target.files)} 
                accept="image/*" 
                style={{ display: 'none' }} 
              />
            </div>
          </div>
        </main>
      ) : (
        <div className="app-layout">
          
          {/* Sidebar */}
          <aside className="sidebar">
            <div className="sidebar-header">
              <span>Session History</span>
              {hasPending && (
                <button className="btn btn-primary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }} onClick={analyzeAllPending}>
                  <Play size={12} /> Analyze All
                </button>
              )}
            </div>
            <div className="sidebar-content">
              {history.map(item => (
                <div 
                  key={item.id} 
                  className={`history-item ${activeId === item.id ? 'active' : ''}`}
                  onClick={() => setActiveId(item.id)}
                >
                  <img src={item.preview} alt={item.name} className="history-thumb" />
                  <div className="history-info">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                      <span className="history-name" title={item.name} style={{ maxWidth: '65%' }}>{item.name}</span>
                      {item.timestamp && (
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      )}
                    </div>
                    <span className="history-status">
                      <span className={`status-badge badge-${item.status}`}></span>
                      {item.status === 'idle' && 'Pending'}
                      {item.status === 'analyzing' && 'Analyzing...'}
                      {item.status === 'done' && (item.predictions[0]?.label || 'Done')}
                      {item.status === 'error' && 'Error'}
                    </span>
                  </div>
                  <button 
                    className="btn-secondary" 
                    style={{ padding: '0.25rem', border: 'none', boxShadow: 'none', background: 'transparent' }} 
                    onClick={(e) => removeItem(item.id, e)}
                    title="Remove item"
                  >
                    <Trash2 size={16} color="var(--text-muted)" />
                  </button>
                </div>
              ))}
            </div>
          </aside>

          {/* Main View */}
          <main className="main-view">
            {activeItem && (
              <div className="dashboard-grid">
                
                {/* Left Column: Image & Actions */}
                <div className="panel" style={{ height: 'fit-content' }}>
                  <div className="image-container">
                    <img src={activeItem.preview} alt="Preview" className="result-image" />
                  </div>
                  
                  <div className="btn-group">
                    {activeItem.status === 'idle' && (
                      <button className="btn btn-primary" onClick={() => analyzeItem(activeItem.id)}>
                        Analyze Image
                      </button>
                    )}
                    
                    {activeItem.status === 'analyzing' && (
                      <button className="btn btn-primary" disabled>
                        <span className="loader"></span> Analyzing...
                      </button>
                    )}

                    {activeItem.status === 'error' && (
                       <button className="btn btn-primary" onClick={() => analyzeItem(activeItem.id)}>
                        <RefreshCcw size={16} /> Retry Analysis
                      </button>
                    )}
                  </div>

                  {activeItem.error && (
                    <div className="error-msg">
                      <AlertCircle size={16} />
                      <span>{activeItem.error}</span>
                    </div>
                  )}
                </div>

                {/* Right Column: Results */}
                {activeItem.predictions.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    
                    {/* Predictions */}
                    <div className="panel">
                      <div className="section-title">Analysis Results</div>
                      <div>
                        {activeItem.predictions.map((pred, i) => (
                          <div key={i} className={`prediction-item ${i === 0 ? 'top-match' : ''}`}>
                            <div className="prediction-header">
                              <span className="prediction-name">{pred.label}</span>
                              <span className="prediction-conf">
                                {pred.conf.toFixed(1)}%
                              </span>
                            </div>
                            <div className="progress-track">
                              <div 
                                className="progress-fill" 
                                style={{ width: `${pred.conf}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Agent Research */}
                    <div className="panel">
                      <div className="agent-header">
                        <BookOpen size={18} className="agent-icon" />
                        <span className="agent-title">Research Assistant</span>
                      </div>
                      
                      <div className="agent-content">
                        {activeItem.status === 'analyzing' && activeItem.predictions.length > 0 ? (
                          <div className="typing-pulse" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span className="loader loader-light"></span>
                            Retrieving data for {activeItem.predictions[0].label}...
                          </div>
                        ) : activeItem.agentResearch ? (
                          <div>
                            {activeItem.agentResearch.title && (
                              <strong>{activeItem.agentResearch.title}</strong>
                            )}
                            <p>{activeItem.agentResearch.summary}</p>
                            {activeItem.agentResearch.url && (
                              <a href={activeItem.agentResearch.url} target="_blank" rel="noreferrer" className="read-more">
                                Read full article <ArrowRight size={14} />
                              </a>
                            )}
                          </div>
                        ) : activeItem.status === 'done' ? (
                           <p>No agent data available.</p>
                        ) : null}
                      </div>
                    </div>

                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}

export default App;
