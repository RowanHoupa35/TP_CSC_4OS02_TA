import time
import random

def bucket_sort(array):
    if len(array) == 0:
        return array

    max_value = max(array) + 1
    bucket_count = 10

    buckets = [[] for _ in range(bucket_count)]

    for num in array:
        index = min(int(num * bucket_count / max_value), bucket_count - 1)
        buckets[index].append(num)
    
    sorted_array = []
    for bucket in buckets:
        for i in range(1, len(bucket)):
            key = bucket[i]
            j = i - 1
            while j >= 0 and bucket[j] > key:
                bucket[j + 1] = bucket[j]
                j -= 1
            bucket[j + 1] = key
        sorted_array.extend(bucket)

    return sorted_array

sizes = [1000, 10000, 50000]

print("BucketSort Séquentiel")

for size in sizes:
    random.seed(42)
    array = [random.randint(0, 100000) for _ in range(size)]

    start_time = time.time()
    sorted_array = bucket_sort(array)
    end_time = time.time()

    print(f"Taille: {size:7d} | Temps: {end_time - start_time:.6f}s")
    
    
    
    
#Rapport :
  #- Le code parallèle est plus lent que le code séquentiel;
  # Loi d'Amdahl : la partie séquentielle (distribution des données, collecte des résultats) domine le temps d'exécution total lorsque le nombre de processus augmente.
  # Pour des tailles de données plus grandes, le code parallèle pourrait devenir plus avantageux.
  #- Le tri par insertion est efficace pour les petits buckets, mais pour des tailles de données plus grandes, des algorithmes de tri plus efficaces (comme le tri rapide ou le tri fusion) pourraient améliorer les performances globales.     
  # Répartition des buckets en parallèle 