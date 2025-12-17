import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. DATOS Y MATRIZ DE DISTANCIAS ---

datos = {
    'punto': list(range(20)), # Crea números del 0 al 19
    'tipo': ['deposito'] + ['cliente'] * 19, 
    'x': [
        0,  # Depósito (0,0)
        2, 6, 8, 3, 9, 5,   # Los originales
        12, 15, 4, 11, 7, 18, 14, 1, 19, 10, 16, 5, 13 # Nuevos
    ],
    'y': [
        0,  # Depósito (0,0)
        7, 4, 9, 1, 2, 8,   # Los originales
        12, 5, 15, 18, 14, 10, 3, 19, 8, 6, 17, 11, 16 # Nuevos
    ]
}
data = pd.DataFrame(datos)
coords = data[['x', 'y']].values

# Matriz de Distancias
diferencias = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
matriz_dist = np.sqrt(np.sum(diferencias**2, axis=-1))

clientes_ids = data[data['tipo'] != 'deposito']['punto'].values
n_clientes = len(clientes_ids)

# Configuración
POBLACION = 100
GENERACIONES = 10
TASA_MUTACION = 0.2

# --- 2. FUNCIONES DE VISUALIZACIÓN (NUEVO) ---

def guardar_mapa_calor(matriz):
    """Genera un heatmap con las distancias entre todos los puntos"""
    plt.figure(figsize=(8, 6))
    plt.imshow(matriz, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distancia')
    
    # Poner los números dentro de los cuadros
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            plt.text(j, i, f'{matriz[i, j]:.1f}', 
                     ha="center", va="center", color="w" if matriz[i, j] < 5 else "k")
            
    plt.title("Mapa de Calor: Distancias entre Puntos")
    plt.xlabel("Punto Destino")
    plt.ylabel("Punto Origen")
    plt.tight_layout()
    plt.savefig('mapa_calor_distancias.png')
    plt.close() # Cerrar para liberar memoria

def guardar_mapa_ruta(ruta_genes, generacion, distancia, df_puntos):
    """Dibuja el mapa con los puntos y las líneas del camino"""
    # 1. Construir la ruta completa con coordenadas (Inicio 0 -> Ruta -> Fin 0)
    ruta_completa = [0] + list(ruta_genes) + [0]
    
    x_coords = []
    y_coords = []
    
    for punto_idx in ruta_completa:
        fila = df_puntos[df_puntos['punto'] == punto_idx]
        x_coords.append(fila['x'].values[0])
        y_coords.append(fila['y'].values[0])
        
    plt.figure(figsize=(8, 6))
    
    # 2. Dibujar las líneas (Camino)
    plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2, label='Ruta')
    
    # 3. Dibujar los puntos (Clientes)
    plt.scatter(df_puntos['x'], df_puntos['y'], c='blue', s=100, label='Clientes')
    
    # 4. Dibujar el Depósito (Diferente color)
    deposito = df_puntos[df_puntos['punto'] == 0]
    plt.scatter(deposito['x'], deposito['y'], c='red', s=150, marker='s', label='Depósito')
    
    # 5. Etiquetas de los puntos (0, 1, 2...)
    for i, row in df_puntos.iterrows():
        plt.text(row['x'], row['y']+0.3, str(row['punto']), fontsize=12, ha='center')
        
    plt.title(f"Generación {generacion} - Mejor Distancia: {distancia:.2f}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Guardar y cerrar
    nombre_archivo = f"ruta_gen_{generacion}.png"
    plt.savefig(nombre_archivo)
    plt.close()

# --- 3. FUNCIONES DEL GENÉTICO (TSP) ---

def inicializar_poblacion(num_individuos):
    pob = []
    for _ in range(num_individuos):
        individuo = np.random.permutation(clientes_ids)
        pob.append(individuo)
    return np.array(pob)

def calcular_distancia_ruta(individuo):
    distancia = 0
    actual = 0 
    for siguiente in individuo:
        distancia += matriz_dist[actual][siguiente]
        actual = siguiente
    distancia += matriz_dist[actual][0]
    return distancia

def evaluar_poblacion(poblacion):
    distancias = np.zeros(len(poblacion))
    for i in range(len(poblacion)):
        distancias[i] = calcular_distancia_ruta(poblacion[i])
    return distancias

def seleccionar_padres(poblacion, distancias):
    padres = []
    for _ in range(len(poblacion)):
        i, j = np.random.randint(0, len(poblacion), size=2)
        if distancias[i] < distancias[j]:
            padres.append(poblacion[i])
        else:
            padres.append(poblacion[j])
    return np.array(padres)

def cruzar_ordenado(padre1, padre2):
    size = len(padre1)
    start, end = sorted(np.random.randint(0, size, size=2))
    hijo = np.full(size, -1)
    hijo[start:end] = padre1[start:end]
    pointer = 0
    for ciudad in padre2:
        if ciudad not in hijo:
            while hijo[pointer] != -1:
                pointer += 1
            hijo[pointer] = ciudad
    return hijo

def crear_nueva_generacion(padres):
    hijos = []
    np.random.shuffle(padres)
    for i in range(0, len(padres), 2):
        hijo1 = cruzar_ordenado(padres[i], padres[i+1])
        hijo2 = cruzar_ordenado(padres[i+1], padres[i])
        hijos.append(hijo1)
        hijos.append(hijo2)
    return np.array(hijos)

def mutar(poblacion, tasa):
    for i in range(len(poblacion)):
        if np.random.rand() < tasa:
            idx1, idx2 = np.random.randint(0, n_clientes, size=2)
            poblacion[i][idx1], poblacion[i][idx2] = poblacion[i][idx2], poblacion[i][idx1]
    return poblacion

# --- 4. EJECUCIÓN PRINCIPAL ---

# 1. Generar Mapa de Calor de Distancias (Solo una vez al principio)
print("Generando mapa de calor de distancias...")
guardar_mapa_calor(matriz_dist)

poblacion = inicializar_poblacion(POBLACION)
historial_mejores = []

print(f"Iniciando evolución ({GENERACIONES} generaciones)...")

for gen in range(GENERACIONES):
    # Evaluar
    distancias = evaluar_poblacion(poblacion)
    
    # Encontrar el MEJOR
    idx_mejor = np.argmin(distancias)
    mejor_distancia = distancias[idx_mejor]
    mejor_ruta = poblacion[idx_mejor].copy()
    
    # --- VISUALIZACIÓN: GUARDAR MAPA DE LA RUTA ---
    # Genera la imagen 'ruta_gen_0.png', 'ruta_gen_1.png', etc.
    guardar_mapa_ruta(mejor_ruta, gen, mejor_distancia, data)
    
    # Guardar datos
    ruta_completa_str = str([0] + list(mejor_ruta) + [0])
    historial_mejores.append({
        'Generacion': gen,
        'Distancia': mejor_distancia,
        'Ruta': ruta_completa_str
    })
    
    print(f"Gen {gen}: Distancia {mejor_distancia:.2f} -> Imagen guardada.")

    # Evolución
    padres = seleccionar_padres(poblacion, distancias)
    hijos = crear_nueva_generacion(padres)
    poblacion = mutar(hijos, TASA_MUTACION)

# --- FIN ---
df_historial = pd.DataFrame(historial_mejores)
df_historial.to_csv('historial_genetico.csv', index=False)

print("\n--- PROCESO TERMINADO ---")
print("1. Revisa 'mapa_calor_distancias.png' para ver la matriz.")
print(f"2. Revisa las {GENERACIONES} imágenes 'ruta_gen_X.png' para ver la evolución.")
print("3. Datos guardados en 'historial_genetico.csv'.")