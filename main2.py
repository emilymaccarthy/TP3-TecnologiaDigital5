import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tarfile
import os
import glob
import os
import gzip
import shutil


def uncompress_and_parse():
    # Ruta del archivo tar
    tar_path = 'ALL_atsp.tar'
    extract_path = 'ALL_atsp_extracted/'

    # Crear el directorio de extracción si no existe
    os.makedirs(extract_path, exist_ok=True)

    # Descomprimir el archivo tar
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
        
    # Obtener todos los archivos .atsp.gz en la carpeta extraída
    file_paths = glob.glob(os.path.join(extract_path, '*.atsp.gz'))
    
    all_results_dfs = []
    
    # Procesar cada archivo .atsp.gz
    for file_path in file_paths:
        # Descomprimir el archivo .atsp.gz
        with gzip.open(file_path, 'rb') as f_in:
            # Crear el nombre de archivo de salida sin la extensión .gz
            out_file_path = file_path[:-3]
            with open(out_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Parsear el archivo .atsp descomprimido para obtener la matriz de distancias
        distance_matrix = parse_atsp_file(out_file_path)
        
        # Simular la ejecución del experimento y generar los resultados
        experiment_results = run_experiment_on_single_matrix(distance_matrix)
        
        # Crear el DataFrame de resultados para este archivo
        results_df = pd.DataFrame(experiment_results)
        
        # Agregar información adicional si es necesario
        results_df['File'] = os.path.basename(file_path)
        
        # Guardar el DataFrame en la lista
        all_results_dfs.append(results_df)
        
        # Eliminar el archivo comprimido .atsp.gz y el archivo .atsp después de leerlo
        os.remove(file_path)
        os.remove(out_file_path)
    
    return all_results_dfs


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
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Encontrar la dimensión de la matriz
        dimension = None
        for line in lines:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
                break
        
        if dimension is None:
            raise ValueError("No se pudo encontrar la dimensión de la matriz en el archivo.")
        
        # Encontrar la sección de la matriz de distancias
        found_section = False
        distance_matrix = []
        for line in lines:
            if found_section and not line.strip().startswith('EOF'):
                # Procesar las líneas de la matriz de distancias hasta 'EOF'
                row = list(map(int, line.split()))
                distance_matrix.append(row)
            elif line.startswith('EDGE_WEIGHT_SECTION'):
                found_section = True
            elif found_section and line.strip().startswith('EOF'):
                # Terminar la lectura al encontrar 'EOF'
                break
        
        if not found_section:
            raise ValueError("No se encontró la sección EDGE_WEIGHT_SECTION en el archivo.")
        
        # Verificar si se leyó correctamente toda la matriz
        if len(distance_matrix) != dimension:
            raise ValueError("La matriz de distancias leída no coincide con la dimensión especificada."
                             f"Se esperaba una dimensión de {dimension}, pero se leyeron {len(distance_matrix)} filas.")
        
        return distance_matrix


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
    # # Ruta del archivo tar
    # tar_path = 'ALL_atsp.tar'
    # extract_path = 'ALL_atsp_extracted/'

    # # # Crear el directorio de extracción si no existe
    # # os.makedirs(extract_path, exist_ok=True)

    # # # Descomprimir el archivo tar
    # # with tarfile.open(tar_path, "r") as tar:
    # #     tar.extractall(path=extract_path)
        
    # # Obtener todos los archivos .atsp en la carpeta extraída
    # file_paths = glob.glob(os.path.join(extract_path, '*.atsp'))

    # # Leer las matrices de distancias de todos los archivos
    # distance_matrices = [parse_atsp_file(file_path) for file_path in file_paths]
    
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
    
    # Llamar a la función para descomprimir y parsear los archivos
    distance_matrices = uncompress_and_parse()
    
    # Ejecutar los experimentos en todas las matrices de distancias cargadas
    all_experiment_results = []
    for distance_matrix in distance_matrices:
        results = run_experiment_on_single_matrix(distance_matrix)
        all_experiment_results.extend(results)

    
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
