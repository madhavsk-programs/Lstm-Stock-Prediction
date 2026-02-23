# 📈 LSTM Stock Price Prediction (Memory-Based Neural Network)

## 🧠 Task 2: Memory-Based Neural Networks for Time-Series Prediction

This project implements a **Long Short-Term Memory (LSTM)** based memory neural network to predict stock prices using historical time-series data. The model is trained on Apple (AAPL) stock data and deployed with a complete pipeline including preprocessing, training, evaluation, and visualization.

---

# 🎯 Problem Statement
The objective of this project is to study and implement a **memory-based neural network** (LSTM) for time-series forecasting.  
Given historical stock price data, the model learns temporal dependencies and predicts future stock prices.

This task satisfies the academic requirement of:
- Implementing RNN/LSTM architecture
- Training on time-series dataset
- Evaluating using regression metrics
- Visualizing predictions

---

# 📊 Dataset Description
- Dataset Used: **Apple (AAPL) Historical Stock Data**
- Source: Kaggle / Financial Time-Series CSV
- Total Records: 1258 rows
- Features: Date, Open, High, Low, Close, Volume, etc.
- Selected Feature: **Close Price** (Univariate Time-Series)

Why Close Price?
> Close price reflects the final market consensus and is widely used in financial forecasting models.

---

# 🏗️ Model Architecture (Memory-Based Neural Network)

The model is built using a **Long Short-Term Memory (LSTM)** network, which is specifically designed to handle sequential and time-dependent data.

### Architecture:

Input Layer (Sequence of 30 timesteps, 1 feature)
↓
LSTM Layer (Hidden Size = 64, Layers = 2)
↓
Fully Connected (Dense) Layer
↓
Output Layer (Next Day Price Prediction)


### Key Configuration:
- Input Size: 1 (Close Price)
- Hidden Size: 64
- Number of Layers: 2
- Sequence Window: 30 Days
- Loss Function: MSELoss
- Optimizer: Adam

---

# ⚙️ Implementation Details

## 1️⃣ Data Preprocessing
The preprocessing pipeline includes:
- Date conversion and chronological sorting (critical for time-series)
- Feature selection (Close price)
- MinMax Normalization (0 to 1 scaling)
- Sliding Window Sequence Creation
- Chronological Train-Test Split (80:20)

Sequence Example:
> Past 30 days → Predict next day stock price

---

## 2️⃣ Sequence Creation (Core Time-Series Logic)
The model does not learn from raw rows directly.  
Instead, we convert data into sequences:
- X shape: (samples, 30, 1)
- y shape: (samples, 1)

This allows the LSTM to learn temporal dependencies and patterns.

---

## 3️⃣ Model Training
- Framework: PyTorch
- Epochs: 20
- Batch Processing: Enabled
- Device: CPU (Compatible with assignment environment)

Training Output:
- Loss decreased consistently across epochs
- Final Loss ≈ 0.001 (scaled domain)

---

# 📉 Evaluation Metrics Used (Regression)

The model is evaluated using standard regression metrics:

| Metric | Description |
|-------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

Example Results:
- MSE: ~990
- RMSE: ~31
- MAE: ~23

Note:
> Higher RMSE is expected due to real-world stock volatility and inverse scaling to original price range.

---

# 📊 Visualization of Predictions
The project generates a comparison graph:
- Actual Stock Prices vs Predicted Prices
- Saved as: `prediction_graph.png`

Visualization is rendered using Matplotlib (Agg backend for compatibility).

---

# 🚀 How to Run the Project (Step-by-Step)

## 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Lstm-Stock-Prediction.git
cd Lstm-Stock-Prediction
📦 Step 2: Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate   # Windows
📥 Step 3: Install Dependencies
pip install -r requirements.txt
🏋️ Step 4: Train the LSTM Model
python train.py

This will:

Load dataset

Preprocess data

Train LSTM model

Save trained model as lstm_model.pth

🔮 Step 5: Generate Predictions
python predict.py
📊 Step 6: Visualize Results
python visualise.py

Output:

prediction_graph.png

📁 Project Structure
LSTM Stock Prediction/
│
├── models/
│   └── lstm_model.py
├── utils/
│   ├── preprocessing.py
│   └── visualisation.py
├── train.py
├── predict.py
├── visualise.py
├── config.py
├── AAPL.csv
├── lstm_model.pth
├── prediction_graph.png
├── requirements.txt
└── README.md
🧪 Technologies Used

Python 3.14

PyTorch (Deep Learning Framework)

NumPy & Pandas (Data Processing)

Scikit-learn (Scaling & Metrics)

Matplotlib (Visualization)

Streamlit (Optional Deployment)

🧠 Key Learning Outcomes

Understanding Memory-Based Neural Networks (LSTM)

Time-Series Data Preprocessing

Sliding Window Sequence Modeling

Regression Evaluation for Forecasting

Model Training using PyTorch

Real-world Financial Data Forecasting

📌 Conclusion

This project successfully demonstrates the implementation of a memory-based neural network (LSTM) for time-series prediction.
The model captures temporal patterns in stock price movements and produces reasonable predictions despite market volatility, fulfilling all academic requirements of the assignment.
