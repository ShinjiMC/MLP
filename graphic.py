import matplotlib.pyplot as plt
import re
import os

def cargar_mse_por_epoch(path):
    epochs = []
    mses = []
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r"Epoch\s+(\d+)\s+-\s+MSE:\s+([0-9.eE+-]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                mses.append(float(match.group(2)))
    return epochs, mses

logs = {
    "XOR": "./output/XOR/log.txt",
    "AND": "./output/AND/log.txt",
    "OR":  "./output/OR/log.txt"
}

colores = {
    "XOR": "red",
    "AND": "green",
    "OR": "orange"
}

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, path) in enumerate(logs.items()):
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        continue

    epochs, mses = cargar_mse_por_epoch(path)
    ax = axs[i]

    ax.plot(epochs, mses, linewidth=3, color=colores[name], label=f"{name} MSE")
    ax.set_title(f"{name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
