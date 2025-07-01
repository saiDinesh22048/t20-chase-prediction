# 🏏 T20 Cricket Chase Prediction using ANN (Flask Web App)

This project is a web-based application built using **Flask** that predicts whether a **T20 cricket chase will be successful** based on real-time match features. It uses a **trained Artificial Neural Network (ANN)** model and a **StandardScaler** to preprocess the input.

---

## 🚀 Features

- Predicts **chase success** or **failure** in a T20 cricket match.
- Interactive web interface for input.
- Displays probability (%) of a successful chase.
- Built with **Flask**, **TensorFlow**, and **scikit-learn**.

---
![WhatsApp Image 2025-07-01 at 08 00 39_ca5fb360](https://github.com/user-attachments/assets/9637c133-c6f3-4653-9eba-e41ce7674308)
![WhatsApp Image 2025-07-01 at 08 00 39_ac5faad8](https://github.com/user-attachments/assets/5fd1f31c-0c1a-47a3-adaa-722298e11118)


## 🧠 Model Info

- Model: Artificial Neural Network (ANN)
- Format: `ann_model.h5` (TensorFlow)
- Scaler: `scaler.pkl` (StandardScaler from scikit-learn)
- Input Features:
  - `Runs From Ball`
  - `Innings Runs`
  - `Innings Wickets`
  - `Balls Remaining`
  - `Target Score`
  - `Total Batter Runs`
  - `Total Non Striker Runs`
  - `Batter Balls Faced`
  - `Non Striker Balls Faced`

---

## 🖥️ How to Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/t20-chase-predictor.git
cd t20-chase-predictor
