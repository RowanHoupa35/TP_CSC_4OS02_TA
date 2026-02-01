import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


# Tags pour les messages MPI
TAG_TASK = 1      # Envoi d'une tache (numero de ligne)
TAG_RESULT = 2    # Envoi du resultat (donnees de la ligne)
TAG_TERMINATE = 3  # Signal de fin


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


def compute_line(mandelbrot_set, y, width, scaleX, scaleY):
    line = np.empty(width, dtype=np.double)
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        line[x] = mandelbrot_set.convergence(c, smooth=True)
    return line


def maitre(comm, nbp, width, height):
    convergence = np.empty((width, height), dtype=np.double)
    num_workers = nbp - 1

    print(f"Mandelbrot MPI - Strategie Maitre-Exécutant")
    print(f"Image : {width} x {height} pixels")
    print(f"Nombre de processus : {nbp} (1 maitre + {num_workers} exécutants)")

    deb = time()

    next_line = 0
    lines_completed = 0

    for worker in range(1, min(nbp, height + 1)):
        if next_line < height:
            comm.send(next_line, dest=worker, tag=TAG_TASK)
            next_line += 1

    while lines_completed < height:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        worker = status.Get_source()

        line_num = result['line']
        line_data = result['data']

        convergence[:, line_num] = line_data
        lines_completed += 1

        if next_line < height:
            comm.send(next_line, dest=worker, tag=TAG_TASK)
            next_line += 1
        else:
            comm.send(-1, dest=worker, tag=TAG_TERMINATE)

    fin = time()

    # Collecter les statistiques de tous les executants
    worker_stats = []
    for worker in range(1, nbp):
        stats = comm.recv(source=worker, tag=TAG_TERMINATE)
        worker_stats.append(stats)

    temps_total = fin - deb
    print(f"Temps total : {temps_total:.4f} s")
    print(f"Lignes calculees : {lines_completed}")

    # Affichage des temps individuels par executant
    print(f"\n--- Temps de calcul par exécutant ---")
    all_times = []
    for i, stats in enumerate(worker_stats):
        t = stats['compute_time']
        n = stats['lines']
        all_times.append(t)
        print(f"  Executant {i+1} : {t:.4f} s ({n} lignes)")

    # Statistiques
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)
    print(f"\nMoyenne     : {mean_time:.4f} s")
    print(f"Ecart-type  : {std_time:.4f} s")
    print(f"Min         : {min(all_times):.4f} s")
    print(f"Max         : {max(all_times):.4f} s")

    return convergence


def executant(comm, mandelbrot_set, width, scaleX, scaleY):
    total_compute_time = 0.0
    lines_processed = 0

    while True:
        status = MPI.Status()
        line_num = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.Get_tag() == TAG_TERMINATE or line_num < 0:
            break

        deb_line = time()
        line_data = compute_line(mandelbrot_set, line_num, width, scaleX, scaleY)
        fin_line = time()

        total_compute_time += (fin_line - deb_line)
        lines_processed += 1

        result = {'line': line_num, 'data': line_data}
        comm.send(result, dest=0, tag=TAG_RESULT)

    # Renvoyer les statistiques au maitre
    stats = {'compute_time': total_compute_time, 'lines': lines_processed}
    comm.send(stats, dest=0, tag=TAG_TERMINATE)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    if nbp < 2:
        if rank == 0:
            print("Erreur : cette strategie necessite au moins 2 processus.")
            print("Utilisation : mpirun -np N python mandelbrot_mpi_maitre_executant.py (N >= 2)")
        return

    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 1024, 1024

    scaleX = 3./width
    scaleY = 2.25/height

    if rank == 0:
        convergence = maitre(comm, nbp, width, height)

        deb_img = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
        fin_img = time()
        print(f"Temps de constitution de l'image : {fin_img-deb_img:.4f} s")

        image.save("mandelbrot_maitre_executant.png")
        print(f"Image sauvegardee : mandelbrot_maitre_executant.png")

    else:
        executant(comm, mandelbrot_set, width, scaleX, scaleY)


if __name__ == "__main__":
    main()
