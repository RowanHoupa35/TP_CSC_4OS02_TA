import numpy as np
from mpi4py import MPI
from time import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    N = 1200

    if N % nbp != 0:
        if rank == 0:
            print(f"Erreur : N ({N}) doit etre divisible par nbp ({nbp})")
        return

    N_loc = N // nbp

    if rank == 0:
        print(f"=== Produit matrice-vecteur par lignes ===")
        print(f"Dimension N = {N}")
        print(f"Nombre de processus = {nbp}")
        print(f"Lignes par processus (N_loc) = {N_loc}")

    row_start = rank * N_loc
    row_end = row_start + N_loc

    comm.Barrier()
    deb = time()
    deb_local = time()

    # Construction de la matrice locale (N_loc lignes x N colonnes)
    A_local = np.array([[(row_start + i + j) % N + 1. for j in range(N)]
                        for i in range(N_loc)], dtype=np.float64)

    # Le vecteur u complet est necessaire pour le produit
    # (chaque processus a besoin de tout u)
    u = np.array([j + 1. for j in range(N)], dtype=np.float64)

    v_local = A_local.dot(u)

    fin_local = time()
    temps_local = fin_local - deb_local

    v = np.zeros(N, dtype=np.float64)
    comm.Allgather(v_local, v)

    comm.Barrier()
    fin = time()

    # Collecter les temps de tous les processus
    all_times = comm.gather(temps_local, root=0)

    if rank == 0:
        temps_total = fin - deb

        print(f"\nTemps total : {temps_total:.6f} s")

        # Verification du resultat (comparaison avec calcul sequentiel)
        A_full = np.array([[(i+j) % N + 1. for j in range(N)] for i in range(N)])
        u_full = np.array([i+1. for i in range(N)])
        v_ref = A_full.dot(u_full)

        erreur = np.linalg.norm(v - v_ref)
        print(f"Erreur (norme) : {erreur:.2e}")

        # Affichage des temps par processus
        print(f"\nTemps de calcul par processus")
        for p, t in enumerate(all_times):
            print(f"  Processus {p} : {t:.6f} s")

        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        print(f"\nMoyenne     : {mean_time:.6f} s")
        print(f"Ecart-type  : {std_time:.6f} s")

        # Affichage partiel du resultat
        print(f"\nv[0:5] = {v[0:5]}")
        print(f"v[-5:] = {v[-5:]}")


if __name__ == "__main__":
    main()
