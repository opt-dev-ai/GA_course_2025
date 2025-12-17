import pandas as pd
import numpy as np

# --- 1. DATOS Y MATRIZ DE DISTANCIAS ---

# Generamos el CSV de puntos (Depósito + Clientes)
datos = {
    'punto': [0, 1, 2, 3, 4, 5, 6],
    'tipo': ['deposito', 'cliente', 'cliente', 'cliente', 'cliente', 'cliente', 'cliente'],
    'x': [0, 2, 6, 8, 3, 9, 5],
    'y': [0, 7, 4, 9, 1, 2, 8]
}
data = pd.DataFrame(datos)
coords = data[['x', 'y']].values

# Calculamos Matriz de Distancias (Vectorizado)
diferencias = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
matriz_dist = np.sqrt(np.sum(diferencias**2, axis=-1))

# Identificar clientes para permutar
clientes_ids = data[data['tipo'] != 'deposito']['punto'].values
n_clientes = len(clientes_ids)

# Configuración
POBLACION = 50
GENERACIONES = 20
TASA_MUTACION = 0.2

# --- 2. FUNCIONES DEL GENÉTICO (TSP) ---

def inicializar_poblacion(num_individuos):
    pob = []
    for _ in range(num_individuos):
        individuo = np.random.permutation(clientes_ids)
        pob.append(individuo)
    return np.array(pob)

def calcular_distancia_ruta(individuo):
    distancia = 0
    actual = 0 # Empezamos en depósito
    for siguiente in individuo:
        distancia += matriz_dist[actual][siguiente]
        actual = siguiente
    distancia += matriz_dist[actual][0] # Vuelta al depósito
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
        if distancias[i] < distancias[j]: # Menor distancia gana
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

# --- 3. EJECUCIÓN CON HISTORIAL ---

poblacion = inicializar_poblacion(POBLACION)

# Lista para guardar los datos
historial_mejores = []

for gen in range(GENERACIONES):
    # 1. Evaluar
    distancias = evaluar_poblacion(poblacion)
    
    # 2. ENCONTRAR EL MEJOR DE ESTA GENERACIÓN
    idx_mejor = np.argmin(distancias)
    mejor_distancia = distancias[idx_mejor]
    mejor_ruta = poblacion[idx_mejor].copy() # .copy() es vital para que no cambie luego
    
    # 3. GUARDAR EN EL HISTORIAL
    # Guardamos la ruta formateada con el 0 al principio y al final para que se entienda
    ruta_completa_str = str([0] + list(mejor_ruta) + [0])
    
    historial_mejores.append({
        'Generacion': gen,
        'Distancia': mejor_distancia,
        'Ruta': ruta_completa_str
    })
    
    # Imprimir por pantalla cada 10
    if gen % 10 == 0:
        print(f"Gen {gen}: Distancia {mejor_distancia:.2f} | Ruta: {ruta_completa_str}")

    # 4. Evolución estándar
    padres = seleccionar_padres(poblacion, distancias)
    hijos = crear_nueva_generacion(padres)
    poblacion = mutar(hijos, TASA_MUTACION)

# --- 4. RESULTADOS Y EXPORTACIÓN ---

# Convertimos la lista de diccionarios a un DataFrame de Pandas
df_historial = pd.DataFrame(historial_mejores)

print("\n--- RESUMEN DE MEJORA ---")
print(df_historial.head()) # Muestra las primeras 5
print("...")
print(df_historial.tail()) # Muestra las últimas 5

# Guardar en CSV
df_historial.to_csv('historial_genetico.csv', index=False)
print("\nHistorial guardado exitosamente en 'historial_genetico.csv'")

# Mostrar el mejor absoluto encontrado en todo el proceso
idx_min_total = df_historial['Distancia'].idxmin()
mejor_fila = df_historial.iloc[idx_min_total]

print(f"\nMEJOR SOLUCIÓN DE TODA LA HISTORIA (Gen {mejor_fila['Generacion']}):")
print(f"Distancia: {mejor_fila['Distancia']:.4f}")
print(f"Camino: {mejor_fila['Ruta']}")