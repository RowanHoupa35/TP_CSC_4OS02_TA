from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()

def bucket_sort_local(array):
    for i in range(1, len(array)):
        key = array[i]
        j = i - 1
        while j >= 0 and array[j] > key:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
    return array

sizes = [1000, 10000, 50000]

if rank == 0:
    print(f"BucketSort MPI - {num_procs} processus")

for array_size in sizes:
    if rank == 0:
        np.random.seed(42)
        array = np.random.randint(0, 100001, size=array_size).tolist()
        max_val = max(array) + 1
    else:
        array = None
        max_val = None

    max_val = comm.bcast(max_val, root=0)

    comm.Barrier()
    start_time = MPI.Wtime()

    if rank == 0:
        chunk_size = array_size // num_procs
        remainder = array_size % num_procs 
        
        chunks = []
        start_idx = 0
        for i in range(num_procs):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(array[start_idx:start_idx + size])
            start_idx += size
    else:
        chunks = None

    local_data = comm.scatter(chunks, root=0)

    
    local_buckets = [[] for _ in range(num_procs)]

    for num in local_data:
        bucket_id = min(int(num * num_procs / max_val), num_procs - 1)
        local_buckets[bucket_id].append(num)


    recv_buckets = comm.alltoall(local_buckets)

    my_bucket = []
    for bucket in recv_buckets:
        my_bucket.extend(bucket)

    local_sorted = bucket_sort_local(my_bucket)

    gathered = comm.gather(local_sorted, root=0)

    if rank == 0:
        sorted_array = []
        for bucket in gathered:
            sorted_array.extend(bucket)

    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Taille: {array_size:7d} | Temps: {end_time - start_time:.6f}s")
