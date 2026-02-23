import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plot_predictions():
    y_test = np.load("y_test_actual.npy")
    predictions = np.load("predictions.npy")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="red")
    plt.title("LSTM Time-Series Prediction (Actual vs Predicted)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("prediction_plot.png")
    plt.show()

    print("Prediction plot saved as prediction_plot.png")