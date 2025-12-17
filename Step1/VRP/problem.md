Dados:
1. Un conjunto de n clientes/ciudades a visitar
2. Una ciudad depósito/base (punto de inicio y fin)
3. Una flota de M camiones/vehículos idénticos
4. Una matriz de distancias d_ij entre cada par de puntos
   (incluyendo el depósito)
5. Cada camión tiene:
   • Capacidad ilimitada (o capacidad Q en versiones más avanzadas)
   • Debe comenzar y terminar en el depósito

Objetivo:
Asignar los clientes a los M camiones y definir las rutas para:
• VISITAR todos los clientes exactamente una vez
• Cada camión regresa al depósito
• MINIMIZAR la distancia total recorrida por toda la flota
• Balancear la carga entre camiones (objetivo secundario)