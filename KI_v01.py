import yaml

# Pfad zur YAML-Datei
yaml_file_path = 'config/default.yaml'

with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Pfad zur Parquet-Datei
parquet_file_path = config['data']['parquet_file_path']

import pandas as pd
import numpy as np



# Parquet-Datei einlesen
df = pd.read_parquet(parquet_file_path)

# DataFrame anzeigen
print(df)

column_names = df.columns
print(column_names)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Teilen Sie die Spalten mit Arrays in separate Spalten auf
df[['P_G_1', 'P_G_2', 'P_G_3', 'P_G_4', 'P_G_5']] = pd.DataFrame(df['P_G'].tolist(), index=df.index)
df[['Q_G_1', 'Q_G_2', 'Q_G_3', 'Q_G_4', 'Q_G_5']] = pd.DataFrame(df['Q_G'].tolist(), index=df.index)
df[['P_L_1', 'P_L_2', 'P_L_3', 'P_L_4', 'P_L_5']] = pd.DataFrame(df['P_L'].tolist(), index=df.index)
df[['Q_L_1', 'Q_L_2', 'Q_L_3', 'Q_L_4', 'Q_L_5']] = pd.DataFrame(df['Q_L'].tolist(), index=df.index)

df[['u_1_real', 'u_2_real', 'u_3_real', 'u_4_real', 'u_5_real']] = pd.DataFrame(df['u_powerfactory_real'].tolist(), index=df.index)
df[['u_1_imag', 'u_2_imag', 'u_3_imag', 'u_4_imag', 'u_5_imag']] = pd.DataFrame(df['u_powerfactory_imag'].tolist(), index=df.index)

# Jetzt haben Sie separate Spalten für jeden Wert in den Arrays

# Wählen Sie die relevanten Spalten für die Transformation aus
X = df[['P_G_1', 'P_G_2', 'P_G_3', 'P_G_4', 'P_G_5',
        'Q_G_1', 'Q_G_2', 'Q_G_3', 'Q_G_4', 'Q_G_5',
        'P_L_1', 'P_L_2', 'P_L_3', 'P_L_4', 'P_L_5',
        'Q_L_1', 'Q_L_2', 'Q_L_3', 'Q_L_4', 'Q_L_5']]

Y = df[['u_1_real', 'u_2_real', 'u_3_real', 'u_4_real', 'u_5_real', 'u_1_imag', 'u_2_imag', 'u_3_imag', 'u_4_imag', 'u_5_imag']]
"""
print(X)
print(Y)
"""

# Normalisieren Sie die Daten mit StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)
# Aufteilen der Daten in Features (X) und Labels (Y)



# Aufteilen der Daten in Trainings-, Validierungs- und Testdatensätze
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Konvertieren der NumPy-Arrays in PyTorch-Tensoren
X_train = torch.Tensor(X_train)
Y_train = torch.Tensor(Y_train)
X_val = torch.Tensor(X_val)
Y_val = torch.Tensor(Y_val)
X_test = torch.Tensor(X_test)
Y_test = torch.Tensor(Y_test)

# Definition eines einfachen neuronalen Netzes mit PyTorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modellinstanz erstellen
model = Net()

# Verlustfunktion und Optimierer definieren
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam Optimizer

# Training des Modells
epochs = config['params-KI']['epochs']

for epoch in range(epochs):
    # Trainingsphase
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    # Validierungsphase
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, Y_val)

    # Ausgabe von Trainings- und Validierungsverlust
    if (epoch + 1) % 10 == 0:  # Zum Beispiel alle 10 Epochen
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}')


# Validierung des Modells
model.eval()
with torch.no_grad():
    val_predictions = model(X_val)

# Berechnen und Anzeigen der Validierungsgenauigkeit (z.B. Mean Squared Error)
val_mse = mean_squared_error(Y_val, val_predictions)
print(f'Validation Mean Squared Error: {val_mse:.8f}')

# Testen des Modells
with torch.no_grad():
    test_predictions = model(X_test)

# Berechnen und Anzeigen der Testgenauigkeit (z.B. Mean Squared Error)
test_mse = mean_squared_error(Y_test, test_predictions)
print(f'Test Mean Squared Error: {test_mse:.8f}')

# Wählen Sie einen zufälligen Fall aus dem Testdatensatz
random_sample_index = np.random.randint(0, len(X_test))
random_sample_X = X_test[random_sample_index]
random_sample_Y_true = Y_test[random_sample_index]
random_sample_Y_pred = test_predictions[random_sample_index]

# Konvertieren Sie die Tensorwerte in NumPy-Arrays für die Grafik
random_sample_X = random_sample_X.numpy()
random_sample_Y_true = random_sample_Y_true.numpy()
random_sample_Y_pred = random_sample_Y_pred.numpy()


import matplotlib.pyplot as plt
# Erstellen und Anzeigen einer Grafik für die Abweichung der Werte für den zufälligen Fall
plt.figure(figsize=(10, 6))
plt.plot(random_sample_Y_true, label='True Values', marker='o')
plt.plot(random_sample_Y_pred, label='Predicted Values', marker='x')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.title('True vs. Predicted Values for a Random Sample')
plt.legend()
plt.grid(True)
plt.show()

# Umkehrtransformation der vorhergesagten Werte in den ursprünglichen Bereich
random_sample_Y_pred_original_scale = scaler.inverse_transform(random_sample_Y_pred.reshape(1, -1))
random_sample_Y_true_original_scale = scaler.inverse_transform(random_sample_Y_true.reshape(1, -1))

# Ausgabe der vorhergesagten Werte im ursprünglichen Bereich
print("Predicted Values (Original Scale):", random_sample_Y_pred_original_scale)
print("True Values (Original Scale):", random_sample_Y_true_original_scale)