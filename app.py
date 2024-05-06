import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def train_network(layers, width, x, y):
    model = nn.Sequential(
        nn.Linear(1, width),
        *[nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(layers - 2)],
        nn.Linear(width, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x_tensor = torch.from_numpy(x).float().view(-1, 1)
    y_tensor = torch.from_numpy(y).float().view(-1, 1)
    
    for _ in range(1000):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    layers = int(data['layers'])
    width = int(data['width'])
    
    # Create a simple sin wave data
    x_train = np.linspace(-10, 10, 400)
    y_train = np.sin(x_train)
    
    model = train_network(layers, width, x_train, y_train)

    x_test = np.linspace(-15, 15, 600)
    y_test = np.sin(x_test)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test, label='Original')
    with torch.no_grad():
        x_test_tensor = torch.from_numpy(x_test).float().view(-1, 1)
        predictions = model(x_test_tensor).numpy()
        plt.plot(x_test, predictions, label='Model Prediction', linestyle='--')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return jsonify({'image': image_base64})

if __name__ == '__main__':
    app.run(debug=True)
