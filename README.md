# 🫀 Cardiac Arrhythmia Detection Using Deep Learning

This project implements a Convolutional Neural Network (CNN) to detect and classify cardiac arrhythmias from ECG signals. The goal is to enable fast, accurate, and real-time ECG analysis for healthcare applications, especially useful for wearable devices and remote monitoring systems.

---

## 🚀 Features

- Classifies ECG beats into five categories:  
  `['F', 'N', 'Q', 'SVEB', 'VEB']`
- Uses CNN architecture for feature extraction from 1D ECG signals
- Integrated real-time R-peak detection using `scipy.signal.find_peaks`
- Visualizations of ECG waveform, detected peaks, and predicted class
- Ready for deployment in clinical settings or wearable tech platforms

---

## 🧠 Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

# Convolutional Block 1
model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(window_size, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Convolutional Block 2
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
