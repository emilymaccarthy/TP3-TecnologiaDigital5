import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tarfile
import os
import glob

# Funciones previamente definidas para las heurísticas y métodos
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

def combined_method(distance_matrix):
    initial_tour = nearest_neighbor(distance_matrix)
    improved_tour = two_opt(initial_tour, distance_matrix)
    return improved_tour

def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    cost += distance_matrix[tour[-1]][tour[0]]
    return cost

def parse_atsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Encontrar la sección de la matriz de distancias
    edge_weight_section_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1
    dimension = int([line.split(":")[1].strip() for line in lines if "DIMENSION" in line][0])
    
    # Leer la matriz de distancias
    matrix_lines = lines[edge_weight_section_index:edge_weight_section_index + dimension]
    matrix = []
    for line in matrix_lines:
        row = list(map(int, line.split()))
        matrix.append(row)

    return np.array(matrix)

# Función para ejecutar los experimentos y recolectar resultados
def run_experiment(distance_matrices):
    results = []
    for matrix in distance_matrices:
        n = len(matrix)
        methods = ["Nearest Neighbor", "Insertion Heuristic", "Combined Method"]
        for method in methods:
            start_time = time.time()
            if method == "Nearest Neighbor":
                tour = nearest_neighbor(matrix)
            elif method == "Insertion Heuristic":
                tour = insertion_heuristic(matrix)
            elif method == "Combined Method":
                tour = combined_method(matrix)
            cost = calculate_tour_cost(tour, matrix)
            elapsed_time = time.time() - start_time
            results.append({
                "Method": method,
                "Number of Cities": n,
                "Cost": cost,
                "Time": elapsed_time
            })
    return results

# Aplicar heurísticas y métodos combinados en la matriz de distancias cargada
def run_experiment_on_single_matrix(distance_matrix):
    results = []
    n = len(distance_matrix)
    methods = ["Nearest Neighbor", "Insertion Heuristic", "Combined Method"]
    for method in methods:
        start_time = time.time()
        if method == "Nearest Neighbor":
            tour = nearest_neighbor(distance_matrix)
        elif method == "Insertion Heuristic":
            tour = insertion_heuristic(distance_matrix)
        elif method == "Combined Method":
            tour = combined_method(distance_matrix)
        cost = calculate_tour_cost(tour, distance_matrix)
        elapsed_time = time.time() - start_time
        results.append({
            "Method": method,
            "Number of Cities": n,
            "Cost": cost,
            "Time": elapsed_time
        })
    return results

# Función principal para experimentar
def main():
    # Ruta del archivo tar
    tar_path = 'ALL_atsp.tar'
    extract_path = 'ALL_atsp_extracted/'

    # Crear el directorio de extracción si no existe
    os.makedirs(extract_path, exist_ok=True)

    # Descomprimir el archivo tar
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
        
    # Obtener todos los archivos .atsp en la carpeta extraída
    file_paths = glob.glob(os.path.join(extract_path, '*.atsp'))

    # Leer las matrices de distancias de todos los archivos
    distance_matrices = [parse_atsp_file(file_path) for file_path in file_paths]
    # Función para generar una matriz de distancia asimétrica aleatoria
    def generate_random_distance_matrix(n, max_distance=100):
        matrix = np.random.randint(1, max_distance, size=(n, n))
        np.fill_diagonal(matrix, 0)  # Las distancias a sí mismas son 0
        return matrix

    # Generar matrices de ejemplo
    distance_matrix_5 = generate_random_distance_matrix(5)
    distance_matrix_10 = generate_random_distance_matrix(10)
    distance_matrix_20 = generate_random_distance_matrix(20)

    # Ejecutar experimentos
    distance_matrices = [distance_matrix_5, distance_matrix_10, distance_matrix_20]
    results = run_experiment(distance_matrices)

    results_df = pd.DataFrame(results)

    # Visualizar resultados
    plt.figure(figsize=(14, 6))

    # Gráfico de costo de los tours
    plt.subplot(1, 2, 1)
    sns.barplot(x="Number of Cities", y="Cost", hue="Method", data=results_df)
    plt.title("Comparación del Costo del Tour")
    plt.xlabel("Número de Ciudades")
    plt.ylabel("Costo del Tour")

    # Gráfico de tiempo de ejecución
    plt.subplot(1, 2, 2)
    sns.barplot(x="Number of Cities", y="Time", hue="Method", data=results_df)
    plt.title("Comparación del Tiempo de Ejecución")
    plt.xlabel("Número de Ciudades")
    plt.ylabel("Tiempo de Ejecución (segundos)")

    plt.tight_layout()
    plt.show()
    
    # Ejecutar los experimentos en todas las matrices de distancias cargadas
    all_experiment_results = []
    for distance_matrix in distance_matrices:
        results = run_experiment_on_single_matrix(distance_matrix)
        all_experiment_results.extend(results)
    # # Ejemplo de uso con el archivo proporcionado
    # file_path = '/mnt/data/ALL_atsp/br17.atsp'  # Actualiza con la ruta correcta
    # distance_matrix = parse_atsp_file(file_path)
    
    # experiment_results = run_experiment_on_single_matrix(distance_matrix)
    # print(experiment_results)
    
    # # Convertir los resultados a un DataFrame para la visualización
    # results_df = pd.DataFrame(experiment_results)
    
    results_df = pd.DataFrame(all_experiment_results)
    # Visualización
    plt.figure(figsize=(14, 6))

    # Gráfico del costo de los tours
    plt.subplot(1, 2, 1)
    sns.barplot(x="Number of Cities", y="Cost", hue="Method", data=results_df)
    plt.title("Comparación del Costo del Tour")
    plt.xlabel("Número de Ciudades")
    plt.ylabel("Costo del Tour")

    # Gráfico del tiempo de ejecución
    plt.subplot(1, 2, 2)
    sns.barplot(x="Number of Cities", y="Time", hue="Method", data=results_df)
    plt.title("Comparación del Tiempo de Ejecución")
    plt.xlabel("Número de Ciudades")
    plt.ylabel("Tiempo de Ejecución (segundos)")

    plt.tight_layout()
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()
