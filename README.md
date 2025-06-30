# Stock Portfolio Prediction System

A comprehensive LSTM-based stock price prediction system for Indian stocks (NSE). This system trains individual LSTM models for each stock and provides accurate price predictions for Open, High, Low, and Close prices.

## 🚀 Features

- **Multi-Stock Training**: Trains individual LSTM models for each stock
- **Comprehensive Predictions**: Predicts Open, High, Low, and Close prices
- **Technical Indicators**: Uses 20+ technical indicators for enhanced predictions
- **Performance Analysis**: Detailed model evaluation with MAE and RMSE metrics
- **Simple Pipeline Interface**: Direct execution of training and prediction pipelines

## 📁 Project Structure

```
stock_portfolio/
├── src/                          # Source code
│   ├── mlproject/               # Core ML components
│   │   ├── data_collection/     # Data collection modules
│   │   ├── config/              # Configuration files
│   │   └── ...                  # Other ML modules
│   └── pipeline/                # Training and prediction pipelines
│       ├── train_pipeline.py    # 🎯 Main training pipeline
│       └── predict_pipeline.py  # 🎯 Main prediction pipeline
├── notebooks/                   # Jupyter notebooks and data
│   └── Data/                   # Data and trained models
│       ├── models/             # Trained LSTM models
│       ├── processed_data/     # Processed stock data
│       └── un_processed_data/  # Raw stock data
├── logs/                       # Timestamped log files
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock_portfolio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   ```

## 📊 Usage

### 1. Training Models

Train LSTM models for all available stocks:

```bash
python src/pipeline/train_pipeline.py
```

**Features:**
- Trains models for all stocks in the dataset
- Uses 20+ technical indicators
- Saves models to `notebooks/Data/models/`
- Generates performance evaluation reports
- Estimated time: 2-4 hours

### 2. Making Predictions

Make predictions using trained models:

```bash
python src/pipeline/predict_pipeline.py
```

**Options:**
- Predict all available stocks
- Predict specific stocks
- Show available stocks

**Example output:**
```
📊 Stock Price Predictions
============================================================
✅ Successful Predictions (3 stocks):
------------------------------------------------------------
📈 RELIANCE.NS:
   💰 Open:   ₹2450.25
   📊 High:   ₹2480.50
   📉 Low:    ₹2430.75
   📋 Close:  ₹2465.80
   🎯 Confidence: 85.2%
```

## 📈 Model Architecture

The system uses a simplified LSTM architecture:

```
Input Layer → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(25) → Output Layer
```

**Key Features:**
- **Lookback Period**: 52 days (1 year of trading data)
- **Target Variables**: Open, High, Low, Close prices
- **Technical Indicators**: 20+ indicators including RSI, MACD, Bollinger Bands
- **Training**: 10 epochs with early stopping
- **Validation**: 10% validation split

## 📊 Performance Metrics

The system evaluates models using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences

**Performance Tiers:**
- **Excellent**: RMSE < 0.1
- **Good**: RMSE 0.1-0.2
- **Fair**: RMSE 0.2-0.3
- **Poor**: RMSE 0.3-0.5
- **Very Poor**: RMSE ≥ 0.5

## 🔧 Configuration

### Data Paths
All data paths are configured in the pipeline files:
- **Training Data**: `notebooks/Data/un_processed_data/stock_price_with_indicators.csv`
- **Models Directory**: `notebooks/Data/models/`
- **Metadata**: `notebooks/Data/models/model_metadata.pkl`

### Model Parameters
Key parameters can be modified in `src/pipeline/train_pipeline.py`:
- `lookback`: Sequence length (default: 52)
- `target_col`: Target variables (default: ['Close', 'Open', 'High', 'Low'])
- `epochs`: Training epochs (default: 10)
- `batch_size`: Batch size (default: 32)

## 📝 Logging

All operations are logged to the `logs/` directory with timestamps:
- Training logs: `logs/YYYY-MM-DD-HH-MM-SS.log`
- Prediction logs: `logs/prediction_YYYY-MM-DD-HH-MM-SS.log`

## 🚨 Troubleshooting

### Common Issues

1. **No trained models found**:
   ```bash
   python src/pipeline/train_pipeline.py
   ```

2. **Memory issues during training**:
   - Reduce batch size in `train_pipeline.py`
   - Train fewer stocks at once

3. **Import errors**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data not found**:
   - Ensure data files exist in `notebooks/Data/`
   - Check file paths in pipeline configuration

### Performance Tips

- **GPU Acceleration**: Install TensorFlow-GPU for faster training
- **Batch Processing**: Use smaller batches if memory is limited
- **Model Selection**: Focus on stocks with good performance metrics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Create an issue with detailed error information

---

**Note**: This system is for educational and research purposes. Always do your own research before making investment decisions. 