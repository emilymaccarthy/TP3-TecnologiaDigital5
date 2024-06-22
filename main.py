import numpy as np
import random
from itertools import permutations
import time

# 1. Heurística Constructiva: Vecino Más Cercano (Nearest Neighbor)
def nearest_neighbor(distance_matrix):
    n = len(distance_matrix)
    visited = [False] * n
    tour = []
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True

    for _ in range(n - 1):
        nearest_city = -1
        min_distance = float('inf')
        for next_city in range(n):
            if not visited[next_city] and distance_matrix[current_city][next_city] < min_distance:
                nearest_city = next_city
                min_distance = distance_matrix[current_city][next_city]
        current_city = nearest_city
        visited[current_city] = True
        tour.append(current_city)
    
    return tour

# 1. Heurística Constructiva: Inserción
def insertion_heuristic(distance_matrix):
    n = len(distance_matrix)
    unvisited = list(range(1, n))
    tour = [0]

    while unvisited:
        best_insertion = None
        min_increase = float('inf')
        for i in range(len(tour)):
            for city in unvisited:
                increase = (distance_matrix[tour[i-1]][city] + distance_matrix[city][tour[i]]) - distance_matrix[tour[i-1]][tour[i]]
                if increase < min_increase:
                    min_increase = increase
                    best_insertion = (i, city)
        
        tour.insert(best_insertion[0], best_insertion[1])
        unvisited.remove(best_insertion[1])

    return tour

# 2. Operador de Búsqueda Local: 2-Opt
def two_opt(tour, distance_matrix):
    n = len(tour)
    best_tour = tour.copy()
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1: continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                if calculate_tour_cost(new_tour, distance_matrix) < calculate_tour_cost(best_tour, distance_matrix):
                    best_tour = new_tour
                    improved = True
        tour = best_tour

    return best_tour

# 2. Operador de Búsqueda Local: Relocation
def relocation(tour, distance_matrix):
    n = len(tour)
    best_tour = tour.copy()
    improved = True

    while improved:
        improved = False
        for i in range(1, n):
            for j in range(n):
                if i == j: continue
                new_tour = tour[:i] + tour[i+1:j+1] + [tour[i]] + tour[j+1:] if i < j else tour[:j] + [tour[i]] + tour[j:i] + tour[i+1:]
                if calculate_tour_cost(new_tour, distance_matrix) < calculate_tour_cost(best_tour, distance_matrix):
                    best_tour = new_tour
                    improved = True
        tour = best_tour

    return best_tour

# 3. Método Combinado: Vecino Más Cercano + 2-Opt
def combined_method(distance_matrix):
    initial_tour = nearest_neighbor(distance_matrix)
    improved_tour = two_opt(initial_tour, distance_matrix)
    return improved_tour

# Función para calcular el costo de un tour
def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    cost += distance_matrix[tour[-1]][tour[0]] # Volver al punto de inicio
    return cost

# Función principal para experimentar
def main():
    # Crear una matriz de distancia de ejemplo (5 ciudades con distancias asimétricas)
    distance_matrix = np.array([
        [0, 10, 15, 20, 25],
        [5, 0, 9, 10, 15],
        [6, 13, 0, 12, 8],
        [8, 8, 9, 0, 11],
        [10, 5, 10, 15, 0]
    ])
    
    # Ejecutar y medir tiempo de heurísticas
    start_time = time.time()
    tour_nn = nearest_neighbor(distance_matrix)
    cost_nn = calculate_tour_cost(tour_nn, distance_matrix)
    time_nn = time.time() - start_time
    
    start_time = time.time()
    tour_insertion = insertion_heuristic(distance_matrix)
    cost_insertion = calculate_tour_cost(tour_insertion, distance_matrix)
    time_insertion = time.time() - start_time

    # Ejecutar y medir tiempo de métodos combinados
    start_time = time.time()
    tour_combined = combined_method(distance_matrix)
    cost_combined = calculate_tour_cost(tour_combined, distance_matrix)
    time_combined = time.time() - start_time

     # Imprimir resultados
    print("Nearest Neighbor Tour: ", tour_nn, " Cost: ", cost_nn, " Time: {:.5f} seconds".format(time_nn))
    print("Insertion Heuristic Tour: ", tour_insertion, " Cost: ", cost_insertion, " Time: {:.5f} seconds".format(time_insertion))
    print("Combined Method Tour: ", tour_combined, " Cost: ", cost_combined, " Time: {:.5f} seconds".format(time_combined))

if __name__ == "__main__":
    main()
