Dados:
1. Un conjunto de n ciudades/puntos a visitar
2. Cada ciudad i tiene:
   • Un tiempo límite (deadline) T_i
3. Una matriz de distancias d_ij entre cada par de ciudades
   (asumimos que distancia = tiempo, o velocidad constante = 1)
4. Un camión que parte de una ciudad base y debe regresar a ella

Objetivo:
Encontrar un recorrido (ruta) que:
• VISITE todas las ciudades exactamente una vez
• REGRESE al punto de inicio
• MINIMICE la distancia total recorrida
• Y llegue a cada ciudad i ANTES de su tiempo límite T_i