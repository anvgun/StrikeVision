# OptionsPulse Analytics

## Overview
OptionsPulse Analytics is a professional-grade options analysis dashboard built on Streamlit that provides sophisticated visualization and interpretation of options market data. The platform enables traders and analysts to gain insights into market sentiment, identify potential price barriers, and analyze options volume distribution across different strike prices.

![image](https://github.com/user-attachments/assets/9884cfde-5039-4c64-b04f-406af3423404)



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

## Usage Guide

### Ticker Selection
Enter a valid stock ticker symbol (e.g., SPY, TSLA, AAPL, NDX, SPX) in the input field. The application supports both standard equity tickers and index tickers.

### Expiry Date Selection
Choose an expiration date from the dropdown menu. The available dates are fetched dynamically based on the selected ticker.

### Strike Range Adjustment
Use the slider to adjust the range of strike prices displayed in the visualization. This allows for focusing on strikes closest to the current price or viewing a wider range of strikes.

### Interpreting the Visualizations

#### Options Volume Heatmap
- **Color intensity**: Higher intensity indicates larger volume at that strike price
- **Current price line**: Red dashed line showing the current price of the underlying asset
- **Expected range**: Shaded area showing the statistically expected trading range based on volume distribution
- **Market sentiment**: Displayed at the top with color coding (green for bullish, red for bearish, orange for neutral)
- **Call/Put ratio**: The ratio between call and put volumes, with higher values indicating bullish sentiment

#### Directional Bias Meter
- **Horizontal meter**: Visualizes the directional bias from bearish (left) to bullish (right)
- **Directional score**: Quantifies the strength of the directional bias from 0-100
- **Market view**: Interpretation of how market participants are positioning (speculative, protective, volatile, or balanced)
- **Capital bias**: Analysis of where capital is being allocated (calls vs puts)
- **Call/Put skew**: Analysis of whether volume is concentrated in OTM or ITM options

#### Delta-Adjusted Open Interest Chart
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

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data provided by Yahoo Finance API
- Built with Streamlit, Plotly, and Pandas
- Inspired by professional options analysis tools used by institutional traders
