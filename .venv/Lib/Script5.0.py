import sys
import pandas as pd
import numpy as np
import requests
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

# ---------------------------
# Auto-Update System
# ---------------------------

CURRENT_VERSION = "1.0.0"  # Update this when releasing new versions
VERSION_URL = "https://raw.githubusercontent.com/Adamj-3x3/version.txt/refs/heads/main/version.txt"
UPDATE_URL = "https://github.com/Adamj-3x3/main/releases/latest/download/yourapp.exe"

def check_for_updates():
    """Checks if a new version is available and downloads it if needed."""
    try:
        response = requests.get(VERSION_URL)
        latest_version = response.text.strip()

        if latest_version > CURRENT_VERSION:
            print(f"New version {latest_version} available! Downloading update...")
            download_update()
        else:
            print("You're up to date!")
    except Exception as e:
        print(f"Error checking for updates: {e}")

def download_update():
    """Downloads the latest version of the application."""
    try:
        response = requests.get(UPDATE_URL, stream=True)
        with open("new_version.exe", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        print("Update downloaded! Restarting app...")
        restart_app()
    except Exception as e:
        print(f"Update failed: {e}")

def restart_app():
    """Restarts the app using the new version."""
    os.system("new_version.exe")  # Runs the new version
    sys.exit()  # Exits the old version

check_for_updates()

# ---------------------------
# Data processing functions
# ---------------------------

# Fix file path handling
BASE_DIR = os.path.dirname(__file__)  # Ensures files are loaded from script location
FILE_M2 = os.path.join(BASE_DIR, 'Global M2.csv')
FILE_BTC = os.path.join(BASE_DIR, 'BTC.csv')

def load_and_filter_csv(filepath, date_col, value_col):
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    return df[[value_col]]

def compute_lag_correlations(df_merged, max_lag):
    correlations = {}
    for lag in range(1, max_lag + 1):  # Ensure lag is never zero
        shifted_btc = df_merged['close_btc'].shift(-lag)
        aligned = pd.concat([df_merged['close_m2'], shifted_btc], axis=1).dropna()
        if not aligned.empty:
            corr = aligned['close_m2'].corr(aligned['close_btc'])
            correlations[lag] = corr
    return correlations

# ---------------------------
# Investment Features
# ---------------------------

STOCKS = {
    "Bitcoin (BTC)": "BTC.csv",
    "Ethereum (ETH)": "ETH.csv",
    "S&P 500": "SP500.csv",
    "Gold (XAU)": "GOLD.csv"
}

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_date, end_date):
        super().__init__()
        self.setWindowTitle("Investment Correlation Analysis")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setGeometry(100, 100, 1024, 768)
        self.start_date = start_date
        self.end_date = end_date
        self.selected_stock = "Bitcoin (BTC)"  # Default stock
        self.initUI()

    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Stock Picker Feature
        self.stock_picker = QtWidgets.QComboBox()
        self.stock_picker.addItems(STOCKS.keys())
        self.stock_picker.currentIndexChanged.connect(self.update_stock)
        layout.addWidget(self.stock_picker)

        # Graph
        self.fig, self.axes = plt.subplots(1, 2, figsize=(18, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Refresh Button
        refresh_btn = QtWidgets.QPushButton("Refresh Analysis")
        refresh_btn.setFixedHeight(40)
        refresh_btn.setStyleSheet("font-size: 16px;")
        refresh_btn.clicked.connect(self.plot_data)
        layout.addWidget(refresh_btn)

        self.plot_data()

    def update_stock(self):
        self.selected_stock = self.stock_picker.currentText()
        self.plot_data()

    def plot_data(self):
        MAX_LAG = 180
        stock_file = os.path.join(BASE_DIR, STOCKS[self.selected_stock])

        df_m2 = load_and_filter_csv(FILE_M2, 'time', 'close').loc[self.start_date:self.end_date]
        df_stock = load_and_filter_csv(stock_file, 'date', 'close').loc[self.start_date:self.end_date]

        df_merged = pd.merge(df_m2.rename(columns={'close': 'close_m2'}),
                             df_stock.rename(columns={'close': 'close_stock'}),
                             left_index=True, right_index=True, how='inner')

        correlations = compute_lag_correlations(df_merged, MAX_LAG)
        optimal_lag = max(correlations, key=correlations.get)
        optimal_corr = correlations[optimal_lag]

        lags = list(correlations.keys())
        corr_values = [correlations[lag] for lag in lags]

        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].plot(lags, corr_values, marker='o', linestyle='-')
        self.axes[0].set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
        self.axes[0].set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
        self.axes[0].set_title(f'Correlation vs. Lag for {self.selected_stock}', fontsize=14, fontweight='bold')
        self.axes[0].axvline(0, color='grey', linestyle='--')
        self.axes[0].grid(True)

        # Moving Average Feature
        optimal_shifted_stock = df_merged['close_stock'].shift(-optimal_lag)
        aligned_optimal = pd.concat([df_merged['close_m2'], optimal_shifted_stock], axis=1).dropna()

        self.axes[1].scatter(aligned_optimal['close_m2'], aligned_optimal['close_stock'], color='blue', alpha=0.6)
        self.axes[1].set_xlabel('Global Money Supply (Close)')
        self.axes[1].set_ylabel(f'{self.selected_stock} Price (Close)')
        self.axes[1].set_title(f'Scatter Plot at Optimal Lag ({optimal_lag} days)')

        # Add Moving Average Line
        if not aligned_optimal.empty:
            aligned_optimal['moving_avg'] = aligned_optimal['close_stock'].rolling(window=14).mean()
            self.axes[1].plot(aligned_optimal['close_m2'], aligned_optimal['moving_avg'], color='red', label='14-day MA')
            self.axes[1].legend()

        self.axes[1].grid(True)
        self.canvas.draw()

# ---------------------------
# Main entry point
# ---------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    start_date, end_date = "2023-01-01", "2024-01-01"  # Default for now
    main_window = MainWindow(start_date, end_date)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
