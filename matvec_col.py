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
        print(f"Produit matrice-vecteur par colonnes")
        print(f"Dimension N = {N}")
        print(f"Nombre de processus = {nbp}")
        print(f"Colonnes par processus (N_loc) = {N_loc}")

    col_start = rank * N_loc
    col_end = col_start + N_loc

    # CONSTRUCTION DE LA PARTIE LOCALE DE LA MATRICE
    # Chaque processus ne stocke que ses N_loc colonnes
    # A_local[i, j_local] = A[i, col_start + j_local] = (i + col_start + j_local) % N + 1
    comm.Barrier()
    deb = time()
    deb_local = time()

    # Construction de la matrice locale (N lignes x N_loc colonnes)
    A_local = np.array([[(i + col_start + j) % N + 1. for j in range(N_loc)]
                        for i in range(N)], dtype=np.float64)

    # Partie locale du vecteur u (seules les composantes col_start:col_end)
    u_local = np.array([col_start + j + 1. for j in range(N_loc)], dtype=np.float64)

    # CALCUL DU PRODUIT PARTIEL
    # v_partial[i] = sum_{j=col_start}^{col_end-1} A[i,j] * u[j]
    v_partial = A_local.dot(u_local)

    fin_local = time()
    temps_local = fin_local - deb_local
    
    # REDUCTION GLOBALE : somme de toutes les contributions partielles
    # Tous les processus obtiennent le resultat complet
    v = np.zeros(N, dtype=np.float64)
    comm.Allreduce(v_partial, v, op=MPI.SUM)

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
