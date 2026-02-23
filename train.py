import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils.preprocessing import TimeSeriesPreprocessor
from models.lstm_model import LSTMModel
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE

def train():
    preprocessor = TimeSeriesPreprocessor()
    preprocessor.load_data()
    preprocessor.preprocess_data()
    X, y = preprocessor.create_sequences()
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel()
    print("\nModel Initialized:")
    print(model)

    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    print("\nTraining Completed!")

    print("\n===== Evaluating Model on Test Data =====")

    model.eval()  

    with torch.no_grad():
        test_predictions = model(X_test)

    y_test_np = y_test.numpy()
    predictions_np = test_predictions.numpy()

    y_test_actual = preprocessor.inverse_transform(y_test_np)
    predictions_actual = preprocessor.inverse_transform(predictions_np)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    mse = mean_squared_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions_actual)

    print("\n===== Regression Evaluation Metrics =====")
    print(f"MSE  (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error): {mae:.4f}")

    np.save("y_test_actual.npy", y_test_actual)
    np.save("predictions.npy", predictions_actual)

    print("\nEvaluation Completed! Predictions saved.")

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model saved as lstm_model.pth")

    return model, preprocessor, X_test, y_test


if __name__ == "__main__":
    train()


