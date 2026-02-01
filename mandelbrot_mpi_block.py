import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


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

        # On verifie dans un premier temps si le complexe
        # n'appartient pas a une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance a la cardioide {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itere
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    mandelbrot_set = MandelbrotSet(max_iterations=1000, escape_radius=2)
    width, height = 1024, 1024

    scaleX = 3./width
    scaleY = 2.25/height

    lines_per_proc = height // nbp
    remainder = height % nbp

    if rank < remainder:
        y_start = rank * (lines_per_proc + 1)
        y_end = y_start + lines_per_proc + 1
    else:
        y_start = rank * lines_per_proc + remainder
        y_end = y_start + lines_per_proc

    local_height = y_end - y_start

    if rank == 0:
        print(f"Image : {width} x {height} pixels")
        print(f"Nombre de processus : {nbp}")
        print(f"Lignes par processus ≈ {lines_per_proc}")

    local_convergence = np.empty((width, local_height), dtype=np.double)

    comm.Barrier()
    deb = time()
    deb_local = time()  # Temps de debut local

    for local_y, y in enumerate(range(y_start, y_end)):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            local_convergence[x, local_y] = mandelbrot_set.convergence(c, smooth=True)

    fin_local = time()  # Temps de fin local (avant barriere)
    temps_local = fin_local - deb_local  # Temps de calcul de CE processus

    comm.Barrier()
    fin_calcul = time()

    # Collecter les temps locaux de tous les processus
    all_times = comm.gather(temps_local, root=0)

    local_data = local_convergence.T.flatten()

    if rank == 0:
        convergence = np.empty((width, height), dtype=np.double)
        recv_data = np.empty(width * height, dtype=np.double)
    else:
        recv_data = None

    sendcounts = np.array([0] * nbp)
    displacements = np.array([0] * nbp)

    for p in range(nbp):
        if p < remainder:
            p_start = p * (lines_per_proc + 1)
            p_lines = lines_per_proc + 1
        else:
            p_start = p * lines_per_proc + remainder
            p_lines = lines_per_proc

        sendcounts[p] = width * p_lines
        displacements[p] = width * p_start

    comm.Gatherv(local_data, [recv_data, sendcounts, displacements, MPI.DOUBLE], root=0)

    fin_comm = time()

    if rank == 0:
        convergence = recv_data.reshape((width, height))

        temps_calcul = fin_calcul - deb
        temps_comm = fin_comm - fin_calcul
        temps_total = fin_comm - deb

        print(f"Temps de calcul (max) : {temps_calcul:.4f} s")
        print(f"Temps de communication : {temps_comm:.4f} s")
        print(f"Temps total : {temps_total:.4f} s")

        # Affichage des temps individuels par processus
        print(f"\n--- Temps de calcul par processus ---")
        for p, t in enumerate(all_times):
            print(f"  Processus {p} : {t:.4f} s")

        # Statistiques
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        print(f"\nMoyenne     : {mean_time:.4f} s")
        print(f"Ecart-type  : {std_time:.4f} s")
        print(f"Min         : {min(all_times):.4f} s")
        print(f"Max         : {max(all_times):.4f} s")

        deb_img = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
        fin_img = time()
        print(f"Temps de constitution de l'image : {fin_img-deb_img:.4f} s")

        image.save("mandelbrot_block.png")
        print(f"Image sauvegardee : mandelbrot_block.png")

        image.show()

if __name__ == "__main__":
    main()
