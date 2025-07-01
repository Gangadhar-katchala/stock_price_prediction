# Stock Price Prediction System

A comprehensive LSTM-based stock price prediction system for Indian stocks (NSE) that trains individual models for each stock and provides accurate predictions for Open, High, Low, and Close prices using advanced technical indicators.

## 🚀 Features

- **Multi-Stock LSTM Models**: Individual LSTM models trained for each stock
- **Comprehensive Predictions**: Predicts Open, High, Low, and Close prices simultaneously
- **Advanced Technical Indicators**: 20+ indicators including RSI, MACD, Bollinger Bands, Ichimoku Cloud
- **Performance Evaluation**: Detailed model evaluation with MAE and RMSE metrics
- **Simple Pipeline Interface**: Easy-to-use training and prediction pipelines
- **Model Persistence**: Saves trained models and metadata for future use

## 📁 Project Structure

```
stock_price_prediction/
├── src/                          # Source code
│   ├── mlproject/               # Core ML components
│   │   ├── data_collection/     # Data collection modules
│   │   ├── model_building.py    # LSTM model architecture and training
│   │   ├── indicators.py        # Technical indicators calculation
│   │   └── logger.py            # Logging configuration
│   └── pipeline/                # Training and prediction pipelines
│       ├── train_pipeline.py    # Main training pipeline
│       └── predict_pipeline.py  # Main prediction pipeline
├── notebooks/                   # Jupyter notebooks and data
│   └── Data/                   # Data and trained models
│       ├── models/             # Trained LSTM models (40+ stocks)
│       ├── processed_data/     # Processed stock data
│       └── un_processed_data/  # Raw stock data with indicators
├── logs/                       # Timestamped log files
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock_price_prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Training Models

Train LSTM models for all available stocks:

```bash
python src/pipeline/train_pipeline.py
```

**Features:**
- Trains individual models for 40+ Indian stocks
- Uses 20+ technical indicators for enhanced predictions
- 52-day lookback period (1 year of trading data)
- Saves models to `notebooks/Data/models/`
- Estimated training time: 2-4 hours

### Making Predictions

Make predictions using trained models:

```bash
python src/pipeline/predict_pipeline.py
```

**Example Output:**
```
📊 Stock Price Predictions
============================================================
✅ Successful Predictions (40 stocks):
------------------------------------------------------------
📈 RELIANCE.NS:
   💰 Open:   ₹2450.25
   📊 High:   ₹2480.50
   📉 Low:    ₹2430.75
   📋 Close:  ₹2465.80
   🎯 Confidence: 85.2%
```

## 🧠 Model Architecture

### LSTM Network Structure
```
Input Layer (52 timesteps, 62 features)
    ↓
LSTM Layer (128 units) + Dropout (0.2)
    ↓
LSTM Layer (64 units) + Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output Layer (4 units: Open, High, Low, Close)
```

### Key Parameters
- **Lookback Period**: 52 days (1 year of trading data)
- **Target Variables**: Open, High, Low, Close prices
- **Feature Count**: 62 features including technical indicators
- **Training**: 10 epochs with early stopping
- **Optimizer**: Adam with learning rate 0.001

## 📈 Technical Indicators

The system uses 20+ technical indicators:

### Moving Averages
- Simple Moving Averages (SMA_20, SMA_50, SMA_100, SMA_200)
- Exponential Moving Averages (EMA_50, EMA_100, EMA_200)

### Momentum Indicators
- Relative Strength Index (RSI)
- Stochastic Oscillator (K%, D%)
- MACD (MACD, Signal, Histogram)

### Volatility Indicators
- Bollinger Bands (Upper, Lower)
- Average True Range (ATR)
- Parabolic SAR

### Support/Resistance
- Pivot Points (Pivot, S1, S2, S3, R1, R2, R3)
- Fibonacci Retracement Levels
- Donchian Channels

### Advanced Indicators
- Ichimoku Cloud (Tenkan, Kijun, Senkou Spans, Chikou)
- On-Balance Volume (OBV)
- Garman-Klass Volatility

## 📊 Performance Metrics

### Evaluation Criteria
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences

### Performance Tiers
- **Excellent**: RMSE < 0.1
- **Good**: RMSE 0.1-0.2
- **Fair**: RMSE 0.2-0.3
- **Poor**: RMSE 0.3-0.5
- **Very Poor**: RMSE ≥ 0.5

### Current Performance
- **Best Performing**: IDEA.NS, YESBANK.NS (RMSE < 1.0)
- **Good Performance**: RELIANCE.NS, TATAMOTORS.NS, CIPLA.NS
- **ETF Performance**: BANKBEES.NS, NIFTYBEES.NS, GOLDBEES.NS

## 🔧 Configuration

### Data Paths
- **Stock List**: `notebooks/Data/processed_data/stock_list.xlsx`
- **Price Data**: `notebooks/Data/processed_data/stock_price_data.csv`
- **Indicators Data**: `notebooks/Data/un_processed_data/stock_price_with_indicators.csv`
- **Models Directory**: `notebooks/Data/models/`
- **Metadata**: `notebooks/Data/models/model_metadata.pkl`

### Model Parameters
Key parameters in `src/pipeline/train_pipeline.py`:
- `lookback`: Sequence length (default: 52)
- `target_col`: Target variables (default: ['Close', 'Open', 'High', 'Low'])
- `epochs`: Training epochs (default: 10)
- `batch_size`: Batch size (default: 32)

## 🚨 Troubleshooting

### Common Issues

1. **No trained models found**:
   ```bash
   python src/pipeline/train_pipeline.py
   ```

2. **Memory issues during training**:
   - Reduce batch size in `train_pipeline.py`
   - Train fewer stocks at once
   - Use GPU acceleration if available

3. **Import errors**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data not found**:
   - Ensure data files exist in `notebooks/Data/`
   - Check file paths in pipeline configuration
   - Verify internet connection for data download

### Performance Tips

- **GPU Acceleration**: Install TensorFlow-GPU for faster training
- **Batch Processing**: Use smaller batches if memory is limited
- **Model Selection**: Focus on stocks with good performance metrics

## 📋 Dependencies

### Core Dependencies
```
tensorflow>=2.8.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
yfinance>=0.1.70
openpyxl>=3.0.0
plotly>=5.0.0
```

## ⚠️ Disclaimer

**Important**: This system is designed for educational and research purposes only. The predictions are based on historical data and technical analysis, which may not accurately predict future market movements.

**Investment Disclaimer**:
- Past performance does not guarantee future results
- Always conduct your own research before making investment decisions
- Consider consulting with financial advisors
- This tool should not be the sole basis for investment decisions

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `logs/` directory
3. Create an issue with detailed error information

---

**Note**: This system represents a sophisticated approach to stock price prediction using machine learning. However, financial markets are inherently unpredictable, and no prediction system can guarantee accurate results. Use this tool responsibly and always combine it with fundamental analysis and professional financial advice. 