import os
import numpy as np
import pandas as pd
import zipfile
import re
from timeit import default_timer as timer


def parse_dat_file(content):
    """Funcție pentru citirea și parsarea datelor din fișierul .dat
    se extrage  instance_name, d, r, SCj, Dk, Cjk
    """
    instance_name = re.search(r'instance_name = "(.*?)";', content).group(1)
    d = int(re.search(r'd = (\d+);', content).group(1))
    r = int(re.search(r'r = (\d+);', content).group(1))
    SCj = [int(x) for x in re.search(r'SCj = \[(.*?)\];', content).group(1).split()]
    Dk = [int(x) for x in re.search(r'Dk = \[(.*?)\];', content).group(1).split()]

    Cjk_str = re.search(r'Cjk = \[(.*?)\];', content, re.DOTALL).group(1)
    rows = []
    current_row = []

    for line in Cjk_str.split('\n'):
        line = line.split('#')[0].strip()
        if not line:
            continue
        line = line.replace('[', '').replace(']', '').strip()
        if line:
            numbers = [int(x) for x in line.split()]
            current_row.extend(numbers)
            if len(current_row) == r:
                rows.append(current_row)
                current_row = []
            elif len(current_row) > r:
                while len(current_row) >= r:
                    rows.append(current_row[:r])
                    current_row = current_row[r:]

    Cjk = np.array(rows)
    return instance_name, d, r, SCj, Dk, Cjk


def find_min_cost_cell(costs, supply, demand, used):
    """Funcție pentru găsirea celulei cu cost minim disponibile

    Parametri:
    - costs: matricea costurilor de transport
    - supply: capacitățile rămase în depozite
    - demand: cererile rămase ale magazinelor
    - used: matrice care marchează celulele deja folosit
    """
    m, n = costs.shape
    min_cost = float('inf')
    min_i = min_j = -1

    # Pentru fiecare linie
    for i in range(m):
        if supply[i] <= 0:
            continue

        # Găsim costul minim pe linia curentă
        # Pentru fiecare depozit (linie) cu capacitate > 0:
        line_costs = [(j, costs[i, j]) for j in range(n) if demand[j] > 0 and not used[i, j]]
        if line_costs:
            # Păstrează coordonatele celulei cu costul cel mai mic găsit
            j, cost = min(line_costs, key=lambda x: x[1])
            if cost < min_cost:
                min_cost = cost
                min_i = i
                min_j = j
    # Returnează: - min_i, min_j: coordonatele celulei cu cost minim disponibilă
    return min_i, min_j


def minimum_cost_method(supply, demand, costs):
    """Implementarea metodei minimului pe linie

    Parametri:
    - supply: vector capacități depozite
    - demand: vector cereri magazine
    - costs: matricea costurilor
    """
    supply = supply.copy()
    demand = demand.copy()
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n), dtype=int)
    iterations = 0
    used = np.zeros((m, n), dtype=bool)
    # Inițializare matrice de alocare și contor iterații
    # Verificăm echilibrul problemei
    total_supply = sum(supply)
    total_demand = sum(demand)

    #  Pentru fiecare iterație
    while True:
        iterations += 1

        # Găsim celula cu cost minim disponibilă
        min_i, min_j = find_min_cost_cell(costs, supply, demand, used)

        if min_i == -1 or min_j == -1:
            break

        # Calculăm cantitatea ce poate fi alocată
        quantity = min(supply[min_i], demand[min_j])

        # Actualizăm alocarea și capacitățile/cererile rămase
        allocation[min_i, min_j] = quantity
        supply[min_i] -= quantity
        demand[min_j] -= quantity
        used[min_i, min_j] = True

        # Verificăm dacă am terminat
        if np.all(supply <= 0) or np.all(demand <= 0):
            break

    return allocation, iterations


def solve_transportation_problem(supply, demand, costs):
    """"Rezolvă problema de transport folosind metoda minimului pe linie

    Proces:
    1. Aplică metoda minimului pe linie pentru alocare
    2. Calculează costul total al soluției
    3. Determină depozitele utilizate (Uj)

    Returnează:
    - allocation: matricea de alocare
    - total_cost: costul total al soluției
    - iterations: numărul de iterații
    - status: statusul rezolvării
    - uj: vector care indică depozitele utilizate

    """
    allocation, iterations = minimum_cost_method(supply, demand, costs)

    # Calculează costul total
    total_cost = int(np.sum(allocation * costs))

    # Calculează Uj (1 if depot is used, 0 if not)
    uj = np.zeros(len(supply), dtype=int)
    for i in range(len(supply)):
        if np.sum(allocation[i]) > 0:
            uj[i] = 1

    return allocation, total_cost, iterations, "Solved", uj


def save_solution(filename, allocation, demand, optimal_cost, iterations, status, uj):
    """Salvează soluția în formatul cerut

    Format fișier output:
    - Xjk: matricea de alocare
    - Uj: vector utilizare depozite (1 folosit, 0 nefolosit)
    - Dk: vector cereri magazine
    - Optim: costul total optim
    - Cost D2R: costul total (identic cu Optim)

    """
    with open(filename, 'w') as f:
        if status == "Solved":
            f.write("Xjk=\t [[")
            for i, row in enumerate(allocation):
                if i > 0:
                    f.write("\t  [")
                f.write(" ".join(map(str, row)))
                if i < len(allocation) - 1:
                    f.write("]\n")
                else:
                    f.write("]]\n")

            f.write("\nUj=\t\t [" + " ".join(map(str, uj)) + "]\n")
            f.write("\nDk=\t\t [" + " ".join(map(str, demand)) + "]\n")
            f.write(f"\nOptim\t= {optimal_cost}\n")
            f.write(f"Cost D2R\t= {optimal_cost}")
        else:
            f.write("No feasible solution found.")


def main():
    # Citește fișierele de input (.dat) din arhivă
    input_path = os.path.join("fisiere_transport", "Lab_simple_instances.zip")
    output_dir = os.path.join("fisiere_transport", "solutie")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    with zipfile.ZipFile(input_path, "r") as zip_ref:
        files = sorted([f for f in zip_ref.namelist() if f.endswith('.dat')],
                       key=lambda x: (
                           "1" if "small" in x else "2" if "medium" in x else "3",
                           int(re.search(r'\d+', x).group())
                       ))

        for file in files:
            try:
                print(f"Processing {file}...")

                content = zip_ref.read(file).decode('utf-8')
                instance_name, d, r, SCj, Dk, Cjk = parse_dat_file(content)

                size = "small" if "small" in file else "medium" if "medium" in file else "large"
                num = re.search(r'(\d+)\.dat$', file).group(1)
                output_name = f"{size}_instance_{int(num):02d}_simple"

                start_time = timer()
                allocation, optimal_cost, iterations, status, uj = solve_transportation_problem(
                    np.array(SCj), np.array(Dk), Cjk)
                end_time = timer()

                # Salvează soluția în format .txt

                solution_filename = os.path.join(output_dir, f"{output_name}.txt")
                save_solution(solution_filename, allocation, Dk, optimal_cost, iterations, status, uj)

                results.append([
                    output_name,
                    optimal_cost,
                    iterations,
                    round(end_time - start_time, 3),
                    status
                ])
                print(f"Solved {output_name}: Cost = {optimal_cost}, Iterations = {iterations}")

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                results.append([
                    output_name if 'output_name' in locals() else file,
                    "-",
                    0,
                    0.0,
                    "Error"
                ])

    # Salvează rezultate in Excel
    df = pd.DataFrame(results, columns=['Instance', 'Optim', 'Iterations', 'Time', 'Status'])
    df.to_excel(os.path.join(output_dir, "Lab_simple_solutions.xlsx"), index=False)

    # Arhivă zip cu toate soluțiile
    with zipfile.ZipFile(os.path.join(output_dir, "Lab_simple_solved.zip"), "w") as zip_out:
        for file in os.listdir(output_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(output_dir, file)
                zip_out.write(file_path, os.path.basename(file_path))


if __name__ == "__main__":
    main()