Dados:
1. Un conjunto de n clientes con:
   • Ubicación geográfica
   • Tiempo límite (deadline) T_i para cada cliente i
2. Un depósito central (punto de inicio/fin)
3. Una flota de camiones idénticos, cada uno con:
   • Capacidad ilimitada (o capacidad Q opcional)
   • Coste fijo C_fijo por camión utilizado
   • Coste variable por kilómetro = c €/km
4. Una matriz de distancias d_ij entre todos los puntos
   (suponemos distancia = tiempo, velocidad = 1)

Objetivo:
Encontrar una asignación de clientes a camiones y rutas que:
1. VISITE todos los clientes exactamente una vez
2. CUMPLA los deadlines T_i de cada cliente
3. MINIMICE el coste total
4. Cada camión comienza y termina en el depósito