import pandas as pd
import numpy as np


cmp = 15
cmv = 20
try:
    df = pd.read_csv('dataset.csv',skipinitialspace=True)
    print(df)
    # Extraer como arrays de NumPy
    PESOS = df['peso'].values
    VOLUMENES = df['volumen'].values
    VALORES = df['valor'].values
    GENES = len(PESOS) 
    
except FileNotFoundError:
    print("Error: No se encontró el archivo CSV en la ruta especificada.")
except KeyError as e:
    print(f"Error: El CSV no tiene la columna {e}")

POB_SIZE = 100    # Tamaño de la población
GENERACIONES = 100
TASA_MUTACION = 0.02

def calcular_fitness(poblacion):
    """
    Calcula el valor total de los paquetes. 
    Si excede el peso o volumen, el fitness es 0 (penalización absoluta).
    """
    # Producto punto para calcular totales de cada individuo (vectorización NumPy)
    total_pesos = poblacion @ PESOS
    total_vols = poblacion @ VOLUMENES
    total_valores = poblacion @ VALORES
    
    # Máscara binaria: True si cumple ambas restricciones
    es_valido = (total_pesos <= cmp) & (total_vols <= cmv)
    
    # Retornamos el valor si es válido, de lo contrario 0
    return np.where(es_valido, total_valores, 0)

def seleccion_torneo(pob, fitness, k=3):
    """Selecciona los mejores individuos comparando grupos de k al azar."""
    seleccionados = np.empty_like(pob)
    for i in range(len(pob)):
        # Elegimos k índices al azar
        aspirantes = np.random.randint(0, len(pob), k)
        # El que tenga mejor fitness de esos k gana
        ganador = aspirantes[np.argmax(fitness[aspirantes])]
        seleccionados[i] = pob[ganador]
    return seleccionados

def cruce_un_punto(padres):
    """Combina los genes de los padres en parejas."""
    hijos = padres.copy()
    for i in range(0, len(padres), 2):
        punto = np.random.randint(1, GENES)
        hijos[i, punto:] = padres[i+1, punto:]
        hijos[i+1, punto:] = padres[i, punto:]
    return hijos

def mutacion(pob, tasa):
    """Invierte bits aleatoriamente según la tasa de mutación."""
    mascara = np.random.rand(*pob.shape) < tasa
    pob[mascara] = 1 - pob[mascara]
    return pob

# --- 3. EJECUCIÓN DEL ALGORITMO ---

# Inicialización aleatoria (0 o 1)
poblacion = np.random.randint(2, size=(POB_SIZE, GENES))

print(f"Iniciando evolución para {GENES} paquetes...\n")

for gen in range(GENERACIONES):
    fitness = calcular_fitness(poblacion)
    
    # Guardar el mejor de la generación (Elitismo simple)
    mejor_idx = np.argmax(fitness)
    mejor_valor = fitness[mejor_idx]
    
    
    print(f"Gen {gen}: Mejor Valor = {mejor_valor}")

    # Evolución
    padres = seleccion_torneo(poblacion, fitness)
    hijos = cruce_un_punto(padres)
    poblacion = mutacion(hijos, TASA_MUTACION)
    
    # Aseguramos que el mejor individuo no se pierda (Elitismo)
    poblacion[0] = padres[mejor_idx]

# --- 4. RESULTADOS FINALES ---
fitness_final = calcular_fitness(poblacion)
mejor_idx = np.argmax(fitness_final)
mejor_solucion = poblacion[mejor_idx]

print("\n" + "="*30)
print("RESULTADO OPTIMIZADO")
print("="*30)
print(f"Paquetes seleccionados (ID): {np.where(mejor_solucion == 1)[0]}")
print(f"Valor Total: {fitness_final[mejor_idx]}")
print(f"Peso Total: {mejor_solucion @ PESOS} / {cmp}")
print(f"Volumen Total: {mejor_solucion @ VOLUMENES} / {cmv}")