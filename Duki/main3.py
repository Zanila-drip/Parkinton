import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Cargar datos
df = pd.read_csv('parkinsons_disease_progression_500.csv')
numeric_cols = ['UPDRS_Score', 'Tremor_Severity', 'Motor_Function',
                'Speech_Difficulty', 'Balance_Problems', 'Age']
corr_matrix = df[numeric_cols].corr()

# Crear figura
fig, ax = plt.subplots(figsize=(10, 8))

# Función para animación
def animate(i):
    ax.clear()
    if i == 0:
        # Paso 1: Heatmap sin anotaciones
        sns.heatmap(corr_matrix, cmap='coolwarm', cbar=False, ax=ax)
        ax.set_title("Paso 1: Correlaciones entre Variables")
    else:
        # Paso 2: Resaltar correlación máxima
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        max_corr = corr_matrix.unstack().sort_values(ascending=False)[1]
        max_vars = corr_matrix.unstack().sort_values(ascending=False).index[1]
        ax.annotate(f"Correlación más fuerte: {max_vars[0]} vs {max_vars[1]} = {max_corr:.2f}",
                    xy=(0.5, -0.1), xycoords='axes fraction', ha='center', color='red')
        ax.set_title("Paso 2: Detectando Patrones Clave")

# Generar animación
ani = FuncAnimation(fig, animate, frames=2, interval=2000, repeat=False)
plt.close()

# Guardar como GIF
ani.save('heatmap_animado.gif', writer='pillow', fps=0.5)