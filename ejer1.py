import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo y reproducibilidad
sns.set_theme(style="whitegrid")
np.random.seed(42)

# 1. Definición de la población base (Distribución Exponencial: fuertemente asimétrica)
lambda_param = 0.5
pop_mean = 1.0 / lambda_param
pop_std = 1.0 / lambda_param
population = np.random.exponential(scale=pop_mean, size=100_000)

# Tamaños de muestra (n) a evaluar y número de experimentos (k)
sample_sizes = [2, 10, 30, 100]
num_samples = 5_000

# 2. Simulación del Teorema del Límite Central (TLC)
sample_means = {}
for n in sample_sizes:
    # Extraemos 'num_samples' muestras de tamaño 'n' y calculamos la media de cada una
    samples = np.random.choice(population, size=(num_samples, n))
    sample_means[n] = samples.mean(axis=1)

# 3. Visualización
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)

# Población original
sns.histplot(population, kde=True, ax=axes[0], color="crimson", stat="density", bins=40)
axes[0].set_title(f"Población Original\n(Exponencial, μ={pop_mean:.1f})", fontweight="bold")
axes[0].set_xlim(0, 10)

# Distribución de medias según el tamaño de n
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
for ax, n, color in zip(axes[1:], sample_sizes, colors):
    sns.histplot(sample_means[n], kde=True, ax=ax, color=color, stat="density", bins=35)
    
    # Línea vertical en la media teórica poblacional
    ax.axvline(pop_mean, color="black", linestyle="--", linewidth=1.5, label="μ teórica")
    
    # Error estándar teórico: SE = sigma / sqrt(n)
    se_teorico = pop_std / np.sqrt(n)
    se_empirico = np.std(sample_means[n])
    
    ax.set_title(f"n = {n}\nSE: {se_empirico:.2f} (Teórico: {se_teorico:.2f})", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()