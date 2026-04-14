import { useState } from 'react'
import './App.css'

function App() {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

  const sustainabilityFields = [
    { key: 'Soil_pH', label: 'Soil pH', type: 'number', step: '0.1', placeholder: '6.5' },
    { key: 'Soil_Moisture', label: 'Soil Moisture (%)', type: 'number', step: '0.1', placeholder: '42.0' },
    { key: 'Temperature_C', label: 'Temperature (C)', type: 'number', step: '0.1', placeholder: '28.4' },
    { key: 'Rainfall_mm', label: 'Rainfall (mm)', type: 'number', step: '0.1', placeholder: '120.0' },
    { key: 'Crop_Type', label: 'Crop Type', type: 'text', placeholder: 'Wheat' },
    { key: 'Fertilizer_Usage_kg', label: 'Fertilizer Usage (kg)', type: 'number', step: '0.1', placeholder: '78.0' },
    { key: 'Pesticide_Usage_kg', label: 'Pesticide Usage (kg)', type: 'number', step: '0.1', placeholder: '18.0' },
    { key: 'Crop_Yield_ton', label: 'Crop Yield (tons)', type: 'number', step: '0.1', placeholder: '5.1' }
  ]

  const marketFields = [
    { key: 'Product', label: 'Product', type: 'text', placeholder: 'Rice' },
    { key: 'Demand_Index', label: 'Demand Index', type: 'number', step: '0.1', placeholder: '74.5' },
    { key: 'Supply_Index', label: 'Supply Index', type: 'number', step: '0.1', placeholder: '53.2' },
    { key: 'Competitor_Price_per_ton', label: 'Competitor Price per Ton', type: 'number', step: '0.1', placeholder: '212.0' },
    { key: 'Economic_Indicator', label: 'Economic Indicator', type: 'number', step: '0.1', placeholder: '66.9' },
    { key: 'Weather_Impact_Score', label: 'Weather Impact Score', type: 'number', step: '0.1', placeholder: '39.7' },
    { key: 'Seasonal_Factor', label: 'Seasonal Factor', type: 'text', placeholder: 'Monsoon' },
    { key: 'Consumer_Trend_Index', label: 'Consumer Trend Index', type: 'number', step: '0.1', placeholder: '70.0' }
  ]

  const [sustainabilityData, setSustainabilityData] = useState({
    Soil_pH: '',
    Soil_Moisture: '',
    Temperature_C: '',
    Rainfall_mm: '',
    Crop_Type: '',
    Fertilizer_Usage_kg: '',
    Pesticide_Usage_kg: '',
    Crop_Yield_ton: ''
  })

  const [marketData, setMarketData] = useState({
    Product: '',
    Demand_Index: '',
    Supply_Index: '',
    Competitor_Price_per_ton: '',
    Economic_Indicator: '',
    Weather_Impact_Score: '',
    Seasonal_Factor: '',
    Consumer_Trend_Index: ''
  })

  const [activeModel, setActiveModel] = useState('sustainability')
  const [results, setResults] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [loading, setLoading] = useState(false)

  const normalizePayload = (data) => {
    const normalized = {}
    Object.entries(data).forEach(([key, value]) => {
      const isNumericValue = value !== '' && !Number.isNaN(Number(value))
      normalized[key] = isNumericValue ? Number(value) : value
    })
    return normalized
  }

  const requestPrediction = async (endpoint, payload) => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(normalizePayload(payload))
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || 'Prediction failed. Please review your input values.')
    }

    return response.json()
  }

  const handleSustainabilitySubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setErrorMessage('')
    try {
      const data = await requestPrediction('/predict/sustainability', sustainabilityData)
      setResults(data)
      setActiveModel('sustainability')
    } catch (error) {
      setErrorMessage(error.message)
    }
    setLoading(false)
  }

  const handleMarketSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setErrorMessage('')
    try {
      const data = await requestPrediction('/predict/market', marketData)
      setResults(data)
      setActiveModel('market')
    } catch (error) {
      setErrorMessage(error.message)
    }
    setLoading(false)
  }

  const renderInputField = (field, value, setState, state) => (
    <label key={field.key} className="inputGroup">
      <span>{field.label}</span>
      <input
        type={field.type}
        step={field.step}
        className="control"
        value={value}
        placeholder={field.placeholder}
        onChange={(e) => setState({
          ...state,
          [field.key]: e.target.value
        })}
        required
      />
    </label>
  )

  return (
    <div className="pageShell">
      <div className="backdropGlow glowOne" />
      <div className="backdropGlow glowTwo" />

      <main className="layout">
        <header className="hero">
          <p className="heroTag">AI AGRI INTELLIGENCE</p>
          <h1>Cultivar Intelligence Studio</h1>
          <p>
            Predict farm sustainability and market price with explainable, practical recommendations
            generated from your inputs.
          </p>
        </header>

        <section className="panelGrid">
          <article className="card formCard">
            <div className="tabs">
              <button
                type="button"
                className={activeModel === 'sustainability' ? 'tab active' : 'tab'}
                onClick={() => setActiveModel('sustainability')}
              >
                Sustainability
              </button>
              <button
                type="button"
                className={activeModel === 'market' ? 'tab active' : 'tab'}
                onClick={() => setActiveModel('market')}
              >
                Market Price
              </button>
            </div>

            {activeModel === 'sustainability' && (
              <form className="fields" onSubmit={handleSustainabilitySubmit}>
                {sustainabilityFields.map((field) => (
                  renderInputField(field, sustainabilityData[field.key], setSustainabilityData, sustainabilityData)
                ))}
                <button type="submit" className="cta" disabled={loading}>
                  {loading ? 'Running model...' : 'Predict Sustainability'}
                </button>
              </form>
            )}

            {activeModel === 'market' && (
              <form className="fields" onSubmit={handleMarketSubmit}>
                {marketFields.map((field) => (
                  renderInputField(field, marketData[field.key], setMarketData, marketData)
                ))}
                <button type="submit" className="cta" disabled={loading}>
                  {loading ? 'Running model...' : 'Predict Market Price'}
                </button>
              </form>
            )}

            {errorMessage && <p className="errorBanner">{errorMessage}</p>}
          </article>

          <article className="card insightCard">
            <h2>Prediction Insight</h2>
            {!results && <p className="emptyState">Run a prediction to see AI analysis and score details here.</p>}

            {results && (
              <>
                <div className="metricRow">
                  <span>Model Output</span>
                  <strong>{Number(results.prediction).toFixed(3)}</strong>
                </div>
                <div className="analysisBox">
                  {results.analysis}
                </div>
              </>
            )}
          </article>
        </section>
      </div>
      </main>
    </div>
  )
}

export default App