import numpy as np
import pandas as pd
import json

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x

def activation_color(a):
    a = float(np.clip(a, 0, 1))
    r = int(255 * a)
    g = int(255 * (1 - a))
    return f"rgb({r},{g},0)"

# --------------------------------------------------
# Visualization
# --------------------------------------------------

def visualize_nn_html(X, W1, b1, W2, b2, max_hidden=25):

    # ---- Forward pass (YOUR NN math) ----
    Z1 = W1 @ X + b1
    A1 = np.maximum(0, Z1)

    Z2 = W2 @ A1 + b2
    expZ = np.exp(Z2 - np.max(Z2))
    A2 = expZ / np.sum(expZ)

    elements = []

    # ---- INPUT LAYER (TINY) ----
    input_idx = np.random.choice(784, 20, replace=False)
    for n, i in enumerate(input_idx):
        elements.append({
            "data": {
                "id": f"I{py(i)}",
                "layer": "input",
                "activation": 0.0,
                "color": "#bbbbbb"
            },
            "position": {
                "x": 0,
                "y": py(n) * 30
            }
        })

    # ---- HIDDEN LAYER (MEDIUM) ----
    hidden_idx = np.argsort(A1[:, 0])[-max_hidden:]
    for n, j in enumerate(hidden_idx):
        elements.append({
            "data": {
                "id": f"H{py(j)}",
                "label": f"{A1[j,0]:.2f}",
                "layer": "hidden",
                "activation": py(A1[j,0]),
                "color": activation_color(A1[j,0])
            },
            "position": {
                "x": 300,
                "y": py(n) * 40
            }
        })

    # ---- OUTPUT LAYER (LARGE) ----
    for k in range(10):
        elements.append({
            "data": {
                "id": f"O{py(k)}",
                "label": f"{k}\n{A2[k,0]:.2f}",
                "layer": "output",
                "activation": py(A2[k,0]),
                "color": activation_color(A2[k,0])
            },
            "position": {
                "x": 600,
                "y": py(k) * 70
            }
        })

    # ---- EDGES: INPUT → HIDDEN ----
    for i in input_idx:
        for j in hidden_idx:
            w = W1[j, i]
            if abs(w) > 0.7:
                elements.append({
                    "data": {
                        "source": f"I{py(i)}",
                        "target": f"H{py(j)}",
                        "weight": py(abs(w))
                    }
                })

    # ---- EDGES: HIDDEN → OUTPUT ----
    for j in hidden_idx:
        for k in range(10):
            w = W2[k, j]
            if abs(w) > 0.7:
                elements.append({
                    "data": {
                        "source": f"H{py(j)}",
                        "target": f"O{py(k)}",
                        "weight": py(abs(w))
                    }
                })

    # --------------------------------------------------
    # HTML + Cytoscape.js
    # --------------------------------------------------

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Neural Network Activation Visualization</title>
<script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>

<style>
  body {{
    font-family: Arial, sans-serif;
    margin: 0;
  }}
  #cy {{
    width: 100%;
    height: 100vh;
  }}
</style>
</head>

<body>
<div id="cy"></div>

<script>
var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: {json.dumps(elements)},
  layout: {{ name: 'preset' }},
  style: [

    {{
      selector: 'node[layer = "input"]',
      style: {{
        'background-color': '#bbbbbb',
        'width': 6,
        'height': 6
      }}
    }},

    {{
      selector: 'node[layer = "hidden"]',
      style: {{
        'background-color': 'data(color)',
        'width': 25,
        'height': 25,
        'label': 'data(label)',
        'font-size': '8px',
        'text-valign': 'center'
      }}
    }},

    {{
      selector: 'node[layer = "output"]',
      style: {{
        'background-color': 'data(color)',
        'width': 60,
        'height': 60,
        'label': 'data(label)',
        'font-size': '10px',
        'font-weight': 'bold',
        'text-valign': 'center'
      }}
    }},

    {{
      selector: 'edge',
      style: {{
        'width': 'mapData(weight, 0, 2, 1, 4)',
        'line-color': '#999'
      }}
    }}
  ]
}});
</script>

</body>
</html>
"""

    with open("mnist_nn_visualization.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Saved mnist_nn_visualization.html — open it in your browser")

# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    # Load trained model
    data = np.load("mnist_weights.npz")
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]

    # Load ONE MNIST sample
    df = pd.read_csv("../data/mnist_test.csv")
    X = df.iloc[0, 1:].values.astype(np.float32).reshape(784, 1) / 255.0

    visualize_nn_html(X, W1, b1, W2, b2)
