## System Architecture

<div align="center">
<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg">
  <!-- System Architecture Diagram -->
  <rect x="50" y="20" width="700" height="410" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="45" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">OptionsPulse Analytics - System Architecture</text>
  
  <!-- Data Sources -->
  <rect x="80" y="70" width="180" height="100" rx="5" fill="#e9ecef" stroke="#343a40" stroke-width="1"/>
  <text x="170" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#343a40">Data Sources</text>
  
  <rect x="100" y="110" width="140" height="25" rx="3" fill="white" stroke="#007bff" stroke-width="1"/>
  <text x="170" y="127" font-family="Arial" font-size="12" text-anchor="middle" fill="#007bff">Yahoo Finance API</text>
  
  <rect x="100" y="145" width="140" height="15" rx="2" fill="white" stroke="#6c757d" stroke-width="1"/>
  <text x="170" y="157" font-family="Arial" font-size="10" text-anchor="middle" fill="#6c757d">Market Data Streams</text>
  
  <!-- Core Engine -->
  <rect x="310" y="70" width="180" height="360" rx="5" fill="#e9ecef" stroke="#343a40" stroke-width="1"/>
  <text x="400" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#343a40">Core Engine</text>
  
  <!-- Data Processing Module -->
  <rect x="330" y="110" width="140" height="80" rx="3" fill="#d1e7dd" stroke="#198754" stroke-width="1"/>
  <text x="400" y="130" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#198754">Data Processing</text>
  <text x="400" y="150" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Options Chain Parsing</text>
  <text x="400" y="165" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Greeks Calculation</text>
  <text x="400" y="180" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Strike Filtering</text>
  
  <!-- Analytics Module -->
  <rect x="330" y="200" width="140" height="100" rx="3" fill="#cfe2ff" stroke="#0d6efd" stroke-width="1"/>
  <text x="400" y="220" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#0d6efd">Analytics</text>
  <text x="400" y="240" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">DAOI Calculation</text>
  <text x="400" y="255" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Market Sentiment</text>
  <text x="400" y="270" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Directional Bias</text>
  <text x="400" y="285" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Expected Range</text>
  
  <!-- Visualization Module -->
  <rect x="330" y="310" width="140" height="100" rx="3" fill="#f8d7da" stroke="#dc3545" stroke-width="1"/>
  <text x="400" y="330" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#dc3545">Visualization</text>
  <text x="400" y="350" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Heatmap Generator</text>
  <text x="400" y="365" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Meter Charts</text>
  <text x="400" y="380" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Bar Charts</text>
  <text x="400" y="395" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Annotations</text>
  
  <!-- User Interface -->
  <rect x="540" y="70" width="180" height="360" rx="5" fill="#e9ecef" stroke="#343a40" stroke-width="1"/>
  <text x="630" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#343a40">User Interface</text>
  
  <!-- Input Controls -->
  <rect x="560" y="110" width="140" height="80" rx="3" fill="#fff3cd" stroke="#ffc107" stroke-width="1"/>
  <text x="630" y="130" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#ffc107">Input Controls</text>
  <text x="630" y="150" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Ticker Selection</text>
  <text x="630" y="165" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Expiry Date Picker</text>
  <text x="630" y="180" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Strike Range Slider</text>
  
  <!-- Display Components -->
  <rect x="560" y="200" width="140" height="210" rx="3" fill="#f8f9fa" stroke="#6c757d" stroke-width="1"/>
  <text x="630" y="220" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#6c757d">Dashboard Components</text>
  
  <!-- Chart Containers -->
  <rect x="570" y="240" width="120" height="30" rx="3" fill="white" stroke="#0d6efd" stroke-width="1"/>
  <text x="630" y="260" font-family="Arial" font-size="10" text-anchor="middle" fill="#0d6efd">Volume Heatmap</text>
  
  <rect x="570" y="280" width="120" height="30" rx="3" fill="white" stroke="#198754" stroke-width="1"/>
  <text x="630" y="300" font-family="Arial" font-size="10" text-anchor="middle" fill="#198754">Directional Bias Meter</text>
  
  <rect x="570" y="320" width="120" height="30" rx="3" fill="white" stroke="#dc3545" stroke-width="1"/>
  <text x="630" y="340" font-family="Arial" font-size="10" text-anchor="middle" fill="#dc3545">DAOI Chart</text>
  
  <rect x="570" y="360" width="120" height="30" rx="3" fill="white" stroke="#6c757d" stroke-width="1"/>
  <text x="630" y="380" font-family="Arial" font-size="10" text-anchor="middle" fill="#6c757d">Interpretation Panel</text>
  
  <!-- Data Flow Lines -->
  <!-- API to Processing -->
  <line x1="260" y1="120" x2="330" y2="150" stroke="#343a40" stroke-width="1.5"/>
  <polygon points="330,150 320,146 322,154" fill="#343a40"/>
  
  <!-- Processing to Analytics -->
  <line x1="400" y1="190" x2="400" y2="200" stroke="#343a40" stroke-width="1.5"/>
  <polygon points="400,200 396,190 404,190" fill="#343a40"/>
  
  <!-- Analytics to Visualization -->
  <line x1="400" y1="300" x2="400" y2="310" stroke="#343a40" stroke-width="1.5"/>
  <polygon points="400,310 396,300 404,300" fill="#343a40"/>
  
  <!-- Visualization to UI -->
  <line x1="470" y1="360" x2="570" y2="360" stroke="#343a40" stroke-width="1.5"/>
  <polygon points="570,360 560,356 560,364" fill="#343a40"/>
  
  <!-- UI Input to Processing -->
  <line x1="560" y1="150" x2="470" y2="150" stroke="#343a40" stroke-width="1.5" stroke-dasharray="5,3"/>
  <polygon points="470,150 480,146 480,154" fill="#343a40"/>
  
  <!-- Key Components Annotation -->
  <rect x="80" y="200" width="180" height="200" rx="5" fill="#f8f9fa" stroke="#343a40" stroke-width="1"/>
  <text x="170" y="220" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#343a40">Key Components</text>
  
  <circle cx="100" cy="250" r="5" fill="#007bff"/>
  <text x="180" y="255" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Streamlit Web App Framework</text>
  
  <circle cx="100" cy="280" r="5" fill="#28a745"/>
  <text x="180" y="285" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Pandas Data Processing</text>
  
  <circle cx="100" cy="310" r="5" fill="#dc3545"/>
  <text x="180" y="315" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Plotly Interactive Charts</text>
  
  <circle cx="100" cy="340" r="5" fill="#ffc107"/>
  <text x="180" y="345" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">NumPy Numerical Processing</text>
  
  <circle cx="100" cy="370" r="5" fill="#6c757d"/>
  <text x="180" y="375" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">YFinance API Integration</text>
</svg>
</div>

OptionsPulse Analytics is built with a modular architecture that separates data processing, analytics, and visualization components. The system architecture follows a data flow pattern where market data is processed through several layers before being presented in the user interface.# OptionsPulse Analytics

<div align="center">
<svg viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Logo Background -->
  <rect x="250" y="30" width="300" height="140" rx="20" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Pulse Line -->
  <path d="M280 100 L310 100 L330 60 L360 140 L390 80 L420 120 L450 100 L470 100 L490 70 L520 100" 
        stroke="#28a745" stroke-width="4" fill="none"/>
  
  <!-- Call and Put Arrows -->
  <path d="M340 130 L365 90 L390 130" stroke="#28a745" stroke-width="3" fill="none"/>
  <path d="M420 70 L445 110 L470 70" stroke="#dc3545" stroke-width="3" fill="none"/>
  
  <!-- App Name -->
  <text x="400" y="170" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle" fill="#343a40">OptionsPulse Analytics</text>
</svg>
</div>

## Overview
OptionsPulse Analytics is a professional-grade options analysis dashboard built on Streamlit that provides sophisticated visualization and interpretation of options market data. The platform enables traders and analysts to gain insights into market sentiment, identify potential price barriers, and analyze options volume distribution across different strike prices.

## Features

### 1. Options Volume Heatmap
- Visual representation of options volume across different strike prices
- Categorization by Call/Put and ITM/ATM/OTM
- Automatic identification of key support/resistance levels
- Market sentiment indicators with visual annotations

### 2. Directional Bias Analysis
- Advanced interpretation of options volume distribution
- Visual meter showing bullish/bearish bias based on delta-adjusted calculations
- Identification of market view (speculative, protective, volatile, or balanced)
- Analysis of capital allocation between calls and puts

### 3. Delta-Adjusted Open Interest Analysis
- Calculation and visualization of delta-adjusted open interest
- Identification of potential option walls that may act as support or resistance
- Call/Put ratio analysis with sentiment interpretation
- Automatic detection of significant strike price levels

### 4. Technical Indicators
- Implied volatility consideration in calculations
- Options Greeks computation (Delta, Gamma, Vanna, Charm, Speed, Vomma)
- Expected price range calculation based on volume distribution
- Customizable strike range selection

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install streamlit pandas numpy plotly yfinance
```

### Running the Application
1. Clone this repository:
```bash
git clone https://github.com/yourusername/optionspulse-analytics.git
cd optionspulse-analytics
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run volume_distributions.py
```

4. The application will be available at http://localhost:8501 in your web browser

<div align="center">
<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Installation Flow Diagram -->
  <rect x="50" y="20" width="700" height="260" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="45" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">OptionsPulse Analytics Installation Flow</text>
  
  <!-- Step 1: Prerequisites -->
  <rect x="100" y="70" width="150" height="60" rx="5" fill="#f1f8ff" stroke="#0366d6" stroke-width="1"/>
  <text x="175" y="95" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#0366d6">Step 1</text>
  <text x="175" y="115" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Install Python 3.7+</text>
  
  <!-- Step 2: Clone Repository -->
  <rect x="325" y="70" width="150" height="60" rx="5" fill="#f1f8ff" stroke="#0366d6" stroke-width="1"/>
  <text x="400" y="95" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#0366d6">Step 2</text>
  <text x="400" y="115" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Clone Repository</text>
  
  <!-- Step 3: Install Dependencies -->
  <rect x="550" y="70" width="150" height="60" rx="5" fill="#f1f8ff" stroke="#0366d6" stroke-width="1"/>
  <text x="625" y="95" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#0366d6">Step 3</text>
  <text x="625" y="115" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">Install Dependencies</text>
  
  <!-- Step 4: Run App -->
  <rect x="325" y="160" width="150" height="60" rx="5" fill="#28a745" stroke="#22863a" stroke-width="1"/>
  <text x="400" y="185" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="white">Step 4</text>
  <text x="400" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Run Application</text>
  
  <!-- Flow Arrows -->
  <line x1="250" y1="100" x2="325" y2="100" stroke="#0366d6" stroke-width="2"/>
  <polygon points="325,100 315,96 315,104" fill="#0366d6"/>
  
  <line x1="475" y1="100" x2="550" y2="100" stroke="#0366d6" stroke-width="2"/>
  <polygon points="550,100 540,96 540,104" fill="#0366d6"/>
  
  <line x1="625" y1="130" x2="625" y2="145" stroke="#0366d6" stroke-width="2"/>
  <line x1="625" y1="145" x2="475" y2="145" stroke="#0366d6" stroke-width="2"/>
  <line x1="475" y1="145" x2="475" y2="160" stroke="#0366d6" stroke-width="2"/>
  <polygon points="475,160 471,150 479,150" fill="#0366d6"/>
  
  <!-- Command Box -->
  <rect x="150" y="240" width="500" height="30" rx="5" fill="#f6f8fa" stroke="#d1d5da" stroke-width="1"/>
  <text x="400" y="260" font-family="Courier New" font-size="12" text-anchor="middle" fill="#24292e">streamlit run volume_distributions.py</text>
</svg>
</div>

## Usage Guide

### Ticker Selection
Enter a valid stock ticker symbol (e.g., SPY, TSLA, AAPL, NDX, SPX) in the input field. The application supports both standard equity tickers and index tickers.

### Expiry Date Selection
Choose an expiration date from the dropdown menu. The available dates are fetched dynamically based on the selected ticker.

### Strike Range Adjustment
Use the slider to adjust the range of strike prices displayed in the visualization. This allows for focusing on strikes closest to the current price or viewing a wider range of strikes.

### Interpreting the Visualizations

#### Options Volume Heatmap

<div align="center">
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Heatmap Background -->
  <rect x="100" y="50" width="600" height="300" rx="5" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Y-axis labels -->
  <text x="90" y="80" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Call ITM</text>
  <text x="90" y="130" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Call ATM</text>
  <text x="90" y="180" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Call OTM</text>
  <text x="90" y="230" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Put OTM</text>
  <text x="90" y="280" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Put ATM</text>
  <text x="90" y="330" font-family="Arial" font-size="12" text-anchor="end" fill="#343a40">Put ITM</text>
  
  <!-- X-axis labels -->
  <text x="150" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">410</text>
  <text x="250" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">420</text>
  <text x="350" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">430</text>
  <text x="450" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">440</text>
  <text x="550" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">450</text>
  <text x="650" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">460</text>
  
  <!-- Axis titles -->
  <text x="400" y="390" font-family="Arial" font-size="14" text-anchor="middle" fill="#343a40">Strike Price</text>
  <text x="50" y="200" font-family="Arial" font-size="14" text-anchor="middle" fill="#343a40" transform="rotate(-90, 50, 200)">Option Category</text>
  
  <!-- Heatmap Cells - Call ITM -->
  <rect x="130" y="60" width="60" height="40" fill="#9cec5b" opacity="0.3"/>
  <rect x="200" y="60" width="60" height="40" fill="#9cec5b" opacity="0.6"/>
  <rect x="270" y="60" width="60" height="40" fill="#9cec5b" opacity="0.9"/>
  <rect x="340" y="60" width="60" height="40" fill="#9cec5b" opacity="0.7"/>
  <rect x="410" y="60" width="60" height="40" fill="#9cec5b" opacity="0.4"/>
  <rect x="480" y="60" width="60" height="40" fill="#9cec5b" opacity="0.2"/>
  <rect x="550" y="60" width="60" height="40" fill="#9cec5b" opacity="0.1"/>
  <rect x="620" y="60" width="60" height="40" fill="#9cec5b" opacity="0.1"/>
  
  <!-- Heatmap Cells - Call ATM -->
  <rect x="130" y="110" width="60" height="40" fill="#9cec5b" opacity="0.2"/>
  <rect x="200" y="110" width="60" height="40" fill="#9cec5b" opacity="0.3"/>
  <rect x="270" y="110" width="60" height="40" fill="#9cec5b" opacity="0.5"/>
  <rect x="340" y="110" width="60" height="40" fill="#9cec5b" opacity="0.8"/>
  <rect x="410" y="110" width="60" height="40" fill="#9cec5b" opacity="0.6"/>
  <rect x="480" y="110" width="60" height="40" fill="#9cec5b" opacity="0.3"/>
  <rect x="550" y="110" width="60" height="40" fill="#9cec5b" opacity="0.2"/>
  <rect x="620" y="110" width="60" height="40" fill="#9cec5b" opacity="0.1"/>
  
  <!-- Heatmap Cells - Call OTM -->
  <rect x="130" y="160" width="60" height="40" fill="#9cec5b" opacity="0.1"/>
  <rect x="200" y="160" width="60" height="40" fill="#9cec5b" opacity="0.2"/>
  <rect x="270" y="160" width="60" height="40" fill="#9cec5b" opacity="0.3"/>
  <rect x="340" y="160" width="60" height="40" fill="#9cec5b" opacity="0.4"/>
  <rect x="410" y="160" width="60" height="40" fill="#9cec5b" opacity="0.7"/>
  <rect x="480" y="160" width="60" height="40" fill="#9cec5b" opacity="0.9"/>
  <rect x="550" y="160" width="60" height="40" fill="#9cec5b" opacity="0.6"/>
  <rect x="620" y="160" width="60" height="40" fill="#9cec5b" opacity="0.3"/>
  
  <!-- Heatmap Cells - Put OTM -->
  <rect x="130" y="210" width="60" height="40" fill="#c71585" opacity="0.3"/>
  <rect x="200" y="210" width="60" height="40" fill="#c71585" opacity="0.4"/>
  <rect x="270" y="210" width="60" height="40" fill="#c71585" opacity="0.7"/>
  <rect x="340" y="210" width="60" height="40" fill="#c71585" opacity="0.9"/>
  <rect x="410" y="210" width="60" height="40" fill="#c71585" opacity="0.5"/>
  <rect x="480" y="210" width="60" height="40" fill="#c71585" opacity="0.3"/>
  <rect x="550" y="210" width="60" height="40" fill="#c71585" opacity="0.2"/>
  <rect x="620" y="210" width="60" height="40" fill="#c71585" opacity="0.1"/>
  
  <!-- Heatmap Cells - Put ATM -->
  <rect x="130" y="260" width="60" height="40" fill="#c71585" opacity="0.1"/>
  <rect x="200" y="260" width="60" height="40" fill="#c71585" opacity="0.3"/>
  <rect x="270" y="260" width="60" height="40" fill="#c71585" opacity="0.5"/>
  <rect x="340" y="260" width="60" height="40" fill="#c71585" opacity="0.8"/>
  <rect x="410" y="260" width="60" height="40" fill="#c71585" opacity="0.6"/>
  <rect x="480" y="260" width="60" height="40" fill="#c71585" opacity="0.4"/>
  <rect x="550" y="260" width="60" height="40" fill="#c71585" opacity="0.2"/>
  <rect x="620" y="260" width="60" height="40" fill="#c71585" opacity="0.1"/>
  
  <!-- Heatmap Cells - Put ITM -->
  <rect x="130" y="310" width="60" height="40" fill="#c71585" opacity="0.1"/>
  <rect x="200" y="310" width="60" height="40" fill="#c71585" opacity="0.1"/>
  <rect x="270" y="310" width="60" height="40" fill="#c71585" opacity="0.2"/>
  <rect x="340" y="310" width="60" height="40" fill="#c71585" opacity="0.5"/>
  <rect x="410" y="310" width="60" height="40" fill="#c71585" opacity="0.7"/>
  <rect x="480" y="310" width="60" height="40" fill="#c71585" opacity="0.8"/>
  <rect x="550" y="310" width="60" height="40" fill="#c71585" opacity="0.6"/>
  <rect x="620" y="310" width="60" height="40" fill="#c71585" opacity="0.4"/>
  
  <!-- Current Price Line -->
  <line x1="410" y1="50" x2="410" y2="350" stroke="red" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="420" y="45" font-family="Arial" font-size="10" fill="red">Current Price</text>
  
  <!-- Expected Range -->
  <rect x="270" y="50" width="210" height="300" fill="green" opacity="0.05" stroke="green" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="280" y="45" font-family="Arial" font-size="10" fill="green">Expected Range</text>
  
  <!-- Volume Numbers -->
  <text x="340" y="85" font-family="Arial" font-size="12" text-anchor="middle" fill="white">842</text>
  <text x="410" y="135" font-family="Arial" font-size="12" text-anchor="middle" fill="black">635</text>
  <text x="480" y="185" font-family="Arial" font-size="12" text-anchor="middle" fill="white">912</text>
  <text x="340" y="235" font-family="Arial" font-size="12" text-anchor="middle" fill="white">756</text>
  <text x="410" y="285" font-family="Arial" font-size="12" text-anchor="middle" fill="black">528</text>
  <text x="480" y="335" font-family="Arial" font-size="12" text-anchor="middle" fill="white">693</text>
  
  <!-- Chart Title -->
  <text x="400" y="25" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">SPY Options Volume by Strike Price (Expiry: 2025-05-16)</text>
  
  <!-- Annotations -->
  <rect x="150" y="10" width="140" height="30" rx="5" fill="white" stroke="green" stroke-width="1"/>
  <text x="220" y="30" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="green">BULLISH 🐂</text>
  
  <rect x="310" y="10" width="180" height="30" rx="5" fill="white" stroke="green" stroke-width="1"/>
  <text x="400" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="green">Range: $430-$450</text>
  
  <rect x="510" y="10" width="150" height="30" rx="5" fill="white" stroke="orange" stroke-width="1"/>
  <text x="585" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="orange">C/P Ratio: 1.32</text>
</svg>
</div>
- **Color intensity**: Higher intensity indicates larger volume at that strike price
- **Current price line**: Red dashed line showing the current price of the underlying asset
- **Expected range**: Shaded area showing the statistically expected trading range based on volume distribution
- **Market sentiment**: Displayed at the top with color coding (green for bullish, red for bearish, orange for neutral)
- **Call/Put ratio**: The ratio between call and put volumes, with higher values indicating bullish sentiment

#### Directional Bias Meter

<div align="center">
<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Meter Background -->
  <rect x="100" y="50" width="600" height="200" rx="5" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">SPY Options Directional Bias (Expiry: 2025-05-16)</text>
  
  <!-- Directional Bias Gauge -->
  <rect x="150" y="90" width="500" height="40" rx="15" fill="#f5f5f5" stroke="#ddd" stroke-width="1"/>
  
  <!-- Gauge Sections -->
  <rect x="150" y="90" width="250" height="40" rx="15" fill="#ffcccc" stroke="none"/>
  <rect x="400" y="90" width="250" height="40" rx="15" fill="#ccffcc" stroke="none"/>
  
  <!-- Center Line -->
  <line x1="400" y1="85" x2="400" y2="135" stroke="#aaa" stroke-width="1"/>
  
  <!-- Position Indicator -->
  <circle cx="480" cy="110" r="15" fill="black"/>
  
  <!-- Scale Labels -->
  <text x="150" y="150" font-family="Arial" font-size="12" text-anchor="middle" fill="red">Bearish</text>
  <text x="400" y="150" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Neutral</text>
  <text x="650" y="150" font-family="Arial" font-size="12" text-anchor="middle" fill="green">Bullish</text>
  
  <!-- Volume Distribution Bar -->
  <rect x="150" y="180" width="100" height="40" fill="#006400" stroke="none"/>
  <rect x="250" y="180" width="70" height="40" fill="#00AA00" stroke="none"/>
  <rect x="320" y="180" width="130" height="40" fill="#90EE90" stroke="none"/>
  <rect x="450" y="180" width="80" height="40" fill="#FFC0CB" stroke="none"/>
  <rect x="530" y="180" width="70" height="40" fill="#CD5C5C" stroke="none"/>
  <rect x="600" y="180" width="50" height="40" fill="#8B0000" stroke="none"/>
  
  <!-- Volume Distribution Labels -->
  <text x="200" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Call ITM</text>
  <text x="285" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Call ATM</text>
  <text x="385" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Call OTM</text>
  <text x="490" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Put OTM</text>
  <text x="565" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Put ATM</text>
  <text x="625" y="205" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Put ITM</text>
  
  <!-- Annotations -->
  <rect x="150" y="240" width="250" height="30" rx="5" fill="white" stroke="green" stroke-width="1"/>
  <text x="275" y="260" font-family="Arial" font-size="12" text-anchor="middle" fill="green">Directional Bias: ▲ MODERATE BULLISH (65/100)</text>
  
  <rect x="410" y="240" width="240" height="30" rx="5" fill="white" stroke="blue" stroke-width="1"/>
  <text x="530" y="260" font-family="Arial" font-size="12" text-anchor="middle" fill="blue">Market View: SPECULATIVE (Betting on upside)</text>
</svg>
</div>
- **Horizontal meter**: Visualizes the directional bias from bearish (left) to bullish (right)
- **Directional score**: Quantifies the strength of the directional bias from 0-100
- **Market view**: Interpretation of how market participants are positioning (speculative, protective, volatile, or balanced)
- **Capital bias**: Analysis of where capital is being allocated (calls vs puts)
- **Call/Put skew**: Analysis of whether volume is concentrated in OTM or ITM options

#### Delta-Adjusted Open Interest Chart

<div align="center">
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Chart Background -->
  <rect x="100" y="50" width="600" height="300" rx="5" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Chart Title -->
  <text x="400" y="30" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">SPY Delta-Adjusted Open Interest (Expiry: 2025-05-16)</text>
  
  <!-- Y-axis -->
  <line x1="100" y1="50" x2="100" y2="350" stroke="#666" stroke-width="1"/>
  <line x1="95" y1="200" x2="105" y2="200" stroke="#666" stroke-width="1"/>
  <text x="90" y="205" font-family="Arial" font-size="12" text-anchor="end" fill="#666">0</text>
  
  <line x1="95" y1="100" x2="105" y2="100" stroke="#666" stroke-width="1"/>
  <text x="90" y="105" font-family="Arial" font-size="12" text-anchor="end" fill="#666">+5000</text>
  
  <line x1="95" y1="300" x2="105" y2="300" stroke="#666" stroke-width="1"/>
  <text x="90" y="305" font-family="Arial" font-size="12" text-anchor="end" fill="#666">-5000</text>
  
  <!-- X-axis -->
  <line x1="100" y1="350" x2="700" y2="350" stroke="#666" stroke-width="1"/>
  
  <line x1="150" y1="345" x2="150" y2="355" stroke="#666" stroke-width="1"/>
  <text x="150" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">410</text>
  
  <line x1="250" y1="345" x2="250" y2="355" stroke="#666" stroke-width="1"/>
  <text x="250" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">420</text>
  
  <line x1="350" y1="345" x2="350" y2="355" stroke="#666" stroke-width="1"/>
  <text x="350" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">430</text>
  
  <line x1="450" y1="345" x2="450" y2="355" stroke="#666" stroke-width="1"/>
  <text x="450" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">440</text>
  
  <line x1="550" y1="345" x2="550" y2="355" stroke="#666" stroke-width="1"/>
  <text x="550" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">450</text>
  
  <line x1="650" y1="345" x2="650" y2="355" stroke="#666" stroke-width="1"/>
  <text x="650" y="370" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">460</text>
  
  <!-- Axis Labels -->
  <text x="400" y="390" font-family="Arial" font-size="14" text-anchor="middle" fill="#666">Strike Price</text>
  <text x="50" y="200" font-family="Arial" font-size="14" text-anchor="middle" fill="#666" transform="rotate(-90, 50, 200)">Delta-Adjusted Open Interest</text>
  
  <!-- Current Price Line -->
  <line x1="430" y1="50" x2="430" y2="350" stroke="#511cf6" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="440" y="70" font-family="Arial" font-size="10" fill="orange">Current Price $430.75</text>
  
  <!-- Call DAOI Bars (positive) -->
  <rect x="130" y="190" width="40" height="10" fill="#9cec5b" opacity="0.7"/>
  <rect x="230" y="180" width="40" height="20" fill="#9cec5b" opacity="0.7"/>
  <rect x="330" y="150" width="40" height="50" fill="#9cec5b" opacity="0.7"/>
  <rect x="430" y="120" width="40" height="80" fill="#9cec5b" opacity="0.7"/>
  <rect x="530" y="160" width="40" height="40" fill="#9cec5b" opacity="0.7"/>
  <rect x="630" y="185" width="40" height="15" fill="#9cec5b" opacity="0.7"/>
  
  <!-- Put DAOI Bars (negative) -->
  <rect x="130" y="200" width="40" height="15" fill="#c71585" opacity="0.7"/>
  <rect x="230" y="200" width="40" height="25" fill="#c71585" opacity="0.7"/>
  <rect x="330" y="200" width="40" height="40" fill="#c71585" opacity="0.7"/>
  <rect x="430" y="200" width="40" height="65" fill="#c71585" opacity="0.7"/>
  <rect x="530" y="200" width="40" height="30" fill="#c71585" opacity="0.7"/>
  <rect x="630" y="200" width="40" height="10" fill="#c71585" opacity="0.7"/>
  
  <!-- Option Walls -->
  <rect x="330" y="150" width="40" height="90" stroke="green" stroke-width="2" stroke-dasharray="3,3" fill="none"/>
  <text x="350" y="135" font-family="Arial" font-size="10" fill="green">Support Wall</text>
  
  <rect x="430" y="120" width="40" height="145" stroke="red" stroke-width="2" stroke-dasharray="3,3" fill="none"/>
  <text x="450" y="105" font-family="Arial" font-size="10" fill="red">Resistance Wall</text>
  
  <!-- Annotations -->
  <rect x="200" y="30" width="400" height="30" rx="5" fill="white" stroke="green" stroke-width="1"/>
  <text x="400" y="50" font-family="Arial" font-size="12" text-anchor="middle" fill="green">Delta-Adjusted OI Ratio (Calls/Puts): 1.45 - BULLISH BIAS</text>
  
  <rect x="150" y="70" width="450" height="20" rx="5" fill="white" stroke="blue" stroke-width="1"/>
  <text x="375" y="85" font-family="Arial" font-size="10" text-anchor="middle" fill="blue">↔️ Price likely range-bound between $430.00 and $440.00</text>
</svg>
</div>
- **Green bars (above)**: Call delta-adjusted open interest
- **Magenta bars (below)**: Put delta-adjusted open interest
- **Current price line**: Blue dashed line showing the current price
- **Option walls**: Significant concentrations of delta-adjusted open interest that may act as support/resistance
- **DAOI ratio**: The ratio between call and put delta-adjusted open interest, with interpretation

## Advanced Features

### Delta-Adjusted Open Interest (DAOI)
DAOI weights the open interest by the option's delta, providing a more accurate representation of potential hedging pressure on the underlying asset. This calculation helps identify significant levels where market makers may need to hedge their positions, potentially influencing price action.

### Directional Bias Calculation
The directional bias is calculated by considering:
- The distribution of volume between calls and puts
- The skew between ITM and OTM options
- The concentration of volume at specific strikes relative to the current price
- The premium allocation between calls and puts

### Expected Range Calculation
The expected trading range is determined by:
- Calculating the volume-weighted average of all strike prices
- Determining the standard deviation of volume distribution
- Setting the range as the average ± one standard deviation
- Color-coding based on the width of the range (tighter ranges in green, wider ranges in red)

<div align="center">
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Visual Guide to DAOI -->
  <rect x="50" y="20" width="700" height="360" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="45" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">Advanced Analytics: Key Metrics Explained</text>
  
  <!-- Delta-Adjusted OI Explanation -->
  <rect x="70" y="70" width="320" height="120" rx="5" fill="#edf8ff" stroke="#0366d6" stroke-width="1"/>
  <text x="230" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#0366d6">Delta-Adjusted Open Interest (DAOI)</text>
  
  <!-- DAOI Formula -->
  <rect x="90" y="100" width="280" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="120" font-family="Courier New" font-size="12" text-anchor="middle" fill="#343a40">DAOI = Open Interest × |Delta|</text>
  
  <!-- DAOI Importance -->
  <rect x="90" y="140" width="280" height="40" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="155" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Indicates potential hedging pressure on</text>
  <text x="230" y="170" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">underlying asset from market makers</text>
  
  <!-- Expected Range Explanation -->
  <rect x="410" y="70" width="320" height="120" rx="5" fill="#f8ffed" stroke="#22863a" stroke-width="1"/>
  <text x="570" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#22863a">Expected Price Range</text>
  
  <!-- Range Formula -->
  <rect x="430" y="100" width="280" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="570" y="120" font-family="Courier New" font-size="12" text-anchor="middle" fill="#343a40">Range = WeightedAvg ± StdDev</text>
  
  <!-- Range Importance -->
  <rect x="430" y="140" width="280" height="40" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="570" y="155" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Statistically probable trading range based</text>
  <text x="570" y="170" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">on volume distribution across strikes</text>
  
  <!-- Directional Bias Explanation -->
  <rect x="70" y="210" width="320" height="150" rx="5" fill="#fff5f5" stroke="#d73a49" stroke-width="1"/>
  <text x="230" y="230" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#d73a49">Directional Bias Score</text>
  
  <!-- Bias Formula -->
  <rect x="90" y="240" width="280" height="50" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="257" font-family="Courier New" font-size="11" text-anchor="middle" fill="#343a40">BiasScore = (Bull - Bear) × Skew Ratio</text>
  <text x="230" y="277" font-family="Courier New" font-size="11" text-anchor="middle" fill="#343a40">Bull = CallOTM + CallATM × 0.5</text>
  
  <!-- Bias Importance -->
  <rect x="90" y="300" width="280" height="50" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="315" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Quantifies market sentiment strength</text>
  <text x="230" y="330" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">by weighting call/put volume distribution</text>
  <text x="230" y="345" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">and money flow (ITM vs OTM positioning)</text>
  
  <!-- Market View Explanation -->
  <rect x="410" y="210" width="320" height="150" rx="5" fill="#fff8e6" stroke="#f9c513" stroke-width="1"/>
  <text x="570" y="230" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#b08800">Market View Classification</text>
  
  <!-- Market View Types -->
  <rect x="430" y="240" width="135" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="497" y="260" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">SPECULATIVE</text>
  
  <rect x="575" y="240" width="135" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="642" y="260" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">PROTECTIVE</text>
  
  <rect x="430" y="280" width="135" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="497" y="300" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">VOLATILE</text>
  
  <rect x="575" y="280" width="135" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="642" y="300" font-family="Arial" font-size="12" text-anchor="middle" fill="#343a40">BALANCED</text>
  
  <!-- Market View Explanation -->
  <rect x="430" y="320" width="280" height="30" rx="3" fill="white" stroke="#ddd" stroke-width="1"/>
  <text x="570" y="340" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Interprets positioning strategy of participants</text>
  
  <!-- Connecting Lines -->
  <line x1="230" y1="195" x2="230" y2="210" stroke="#d73a49" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="570" y1="195" x2="570" y2="210" stroke="#f9c513" stroke-width="1" stroke-dasharray="3,3"/>
</svg>
</div>

## Requirements.txt

To help users quickly set up the required dependencies, here's the content for a requirements.txt file:

```
streamlit==1.22.0
pandas==1.5.3
numpy==1.24.3
plotly==5.14.1
yfinance==0.2.18
python-dateutil==2.8.2
requests==2.28.2
```

## Troubleshooting

### Common Issues
- **No data available**: Ensure the ticker symbol is valid and has options trading available
- **Empty visualizations**: Check if the selected expiration date has sufficient trading volume
- **Error fetching options**: Network issues may occur when fetching data from Yahoo Finance; try again later
- **Missing Greeks data**: Some options data may be missing Greeks information; the application will estimate values

### Performance Optimization
- The application uses caching to improve performance when fetching options data
- For tickers with many strike prices, adjust the strike range slider to focus on relevant prices
- Data is refreshed automatically when changing tickers or expiration dates

<div align="center">
<svg viewBox="0 0 800 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Troubleshooting Guide Diagram -->
  <rect x="50" y="20" width="700" height="260" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Title -->
  <text x="400" y="45" font-family="Arial" font-size="16" font-weight="bold" text-anchor="middle" fill="#343a40">OptionsPulse Analytics Troubleshooting Guide</text>
  
  <!-- Data Issues -->
  <rect x="80" y="70" width="180" height="190" rx="5" fill="#f8d7da" stroke="#dc3545" stroke-width="1"/>
  <text x="170" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#dc3545">Data Issues</text>
  
  <!-- Data Issues Types -->
  <rect x="100" y="110" width="140" height="30" rx="3" fill="white" stroke="#dc3545" stroke-width="1"/>
  <text x="170" y="130" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Invalid Ticker Symbol</text>
  
  <rect x="100" y="150" width="140" height="30" rx="3" fill="white" stroke="#dc3545" stroke-width="1"/>
  <text x="170" y="170" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">No Options Available</text>
  
  <rect x="100" y="190" width="140" height="30" rx="3" fill="white" stroke="#dc3545" stroke-width="1"/>
  <text x="170" y="210" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">API Connection Error</text>
  
  <!-- Data Issues Solution -->
  <rect x="100" y="230" width="140" height="20" rx="3" fill="#dc3545" stroke="none" opacity="0.2"/>
  <text x="170" y="245" font-family="Arial" font-size="9" font-style="italic" text-anchor="middle" fill="#dc3545">Try different ticker or check internet</text>
  
  <!-- Visualization Issues -->
  <rect x="310" y="70" width="180" height="190" rx="5" fill="#cfe2ff" stroke="#0d6efd" stroke-width="1"/>
  <text x="400" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#0d6efd">Visualization Issues</text>
  
  <!-- Visualization Issues Types -->
  <rect x="330" y="110" width="140" height="30" rx="3" fill="white" stroke="#0d6efd" stroke-width="1"/>
  <text x="400" y="130" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Empty Charts</text>
  
  <rect x="330" y="150" width="140" height="30" rx="3" fill="white" stroke="#0d6efd" stroke-width="1"/>
  <text x="400" y="170" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Chart Rendering Slow</text>
  
  <rect x="330" y="190" width="140" height="30" rx="3" fill="white" stroke="#0d6efd" stroke-width="1"/>
  <text x="400" y="210" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Inaccurate Analysis</text>
  
  <!-- Visualization Issues Solution -->
  <rect x="330" y="230" width="140" height="20" rx="3" fill="#0d6efd" stroke="none" opacity="0.2"/>
  <text x="400" y="245" font-family="Arial" font-size="9" font-style="italic" text-anchor="middle" fill="#0d6efd">Adjust strike range or expiry date</text>
  
  <!-- Performance Issues -->
  <rect x="540" y="70" width="180" height="190" rx="5" fill="#d1e7dd" stroke="#198754" stroke-width="1"/>
  <text x="630" y="90" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#198754">Performance Tips</text>
  
  <!-- Performance Tips -->
  <rect x="560" y="110" width="140" height="30" rx="3" fill="white" stroke="#198754" stroke-width="1"/>
  <text x="630" y="130" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Use Caching</text>
  
  <rect x="560" y="150" width="140" height="30" rx="3" fill="white" stroke="#198754" stroke-width="1"/>
  <text x="630" y="170" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Limit Strike Range</text>
  
  <rect x="560" y="190" width="140" height="30" rx="3" fill="white" stroke="#198754" stroke-width="1"/>
  <text x="630" y="210" font-family="Arial" font-size="11" text-anchor="middle" fill="#343a40">Select Specific Expiries</text>
  
  <!-- Performance Best Practice -->
  <rect x="560" y="230" width="140" height="20" rx="3" fill="#198754" stroke="none" opacity="0.2"/>
  <text x="630" y="245" font-family="Arial" font-size="9" font-style="italic" text-anchor="middle" fill="#198754">Focus on most liquid options</text>
</svg>
</div>

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data provided by Yahoo Finance API
- Built with Streamlit, Plotly, and Pandas
- Inspired by professional options analysis tools used by institutional traders

<div align="center">
<svg viewBox="0 0 800 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Footer Banner -->
  <rect x="100" y="20" width="600" height="160" rx="10" fill="#f8f9fa" stroke="#343a40" stroke-width="2"/>
  
  <!-- Logo -->
  <rect x="150" y="45" width="140" height="110" rx="10" fill="#e9ecef" stroke="#343a40" stroke-width="1"/>
  
  <!-- Pulse Line in Logo -->
  <path d="M170 100 L190 100 L200 80 L220 120 L230 90 L250 100 L270 100" 
        stroke="#28a745" stroke-width="3" fill="none"/>
        
  <!-- Call and Put Arrows -->
  <path d="M195 110 L210 90 L225 110" stroke="#28a745" stroke-width="2" fill="none"/>
  <path d="M235 80 L250 100 L265 80" stroke="#dc3545" stroke-width="2" fill="none"/>
  
  <!-- App Name in Logo -->
  <text x="220" y="140" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="#343a40">OptionsPulse Analytics</text>
  
  <!-- Tech Stack -->
  <rect x="350" y="45" width="300" height="110" rx="5" fill="white" stroke="#343a40" stroke-width="1"/>
  <text x="500" y="65" font-family="Arial" font-size="14" font-weight="bold" text-anchor="middle" fill="#343a40">Built With</text>
  
  <!-- Tech Icons (simulated) -->
  <circle cx="380" cy="95" r="15" fill="#ff4b4b"/>
  <text x="380" y="100" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="white">S</text>
  <text x="380" y="120" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Streamlit</text>
  
  <circle cx="430" cy="95" r="15" fill="#3366cc"/>
  <text x="430" y="100" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="white">P</text>
  <text x="430" y="120" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Pandas</text>
  
  <circle cx="480" cy="95" r="15" fill="#119DFF"/>
  <text x="480" y="100" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="white">P</text>
  <text x="480" y="120" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">Plotly</text>
  
  <circle cx="530" cy="95" r="15" fill="#013243"/>
  <text x="530" y="100" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="white">N</text>
  <text x="530" y="120" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">NumPy</text>
  
  <circle cx="580" cy="95" r="15" fill="#6610f2"/>
  <text x="580" y="100" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" fill="white">Y</text>
  <text x="580" y="120" font-family="Arial" font-size="10" text-anchor="middle" fill="#343a40">YFinance</text>
  
  <!-- Copyright Notice -->
  <text x="400" y="170" font-family="Arial" font-size="10" text-anchor="middle" fill="#6c757d">© 2025 OptionsPulse Analytics - Advanced Options Trading Analysis Platform</text>
</svg>
</div>
