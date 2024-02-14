import torch
import torch.nn as nn
import torch.optim as optim

# Datos de ejemplo
X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# Modelo de regresión lineal
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Función de costo (Mean Squared Error)
criterion = nn.MSELoss()

# Modelo y optimizador
model = LinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Calcular la pérdida
    loss = criterion(y_pred, y)

    # Backward pass y actualización de parámetros
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluación del modelo entrenado
with torch.no_grad():
    new_data = torch.tensor([[4.0]])
    prediction = model(new_data)
    print(f'Predicción para x=4: {prediction.item()}')
