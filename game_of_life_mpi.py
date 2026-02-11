import numpy as np
import time
import sys
from mpi4py import MPI

class Grille:

    def __init__(self, dim, init_pattern=None):
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)

    def _apply_rules(self, i, j, nb_voisines_vivantes):
        if self.cells[i, j] == 1:  # Cellule vivante
            if nb_voisines_vivantes < 2 or nb_voisines_vivantes > 3:
                self._next_cells[i, j] = 0  # Sous-population ou sur-population
            else:
                self._next_cells[i, j] = 1  # Survie
        elif nb_voisines_vivantes == 3:  # Cellule morte avec 3 voisines vivantes
            self._next_cells[i, j] = 1  # Reproduction
        else:
            self._next_cells[i, j] = 0  # Reste morte

    def prepare_computation(self):
        self._next_cells = np.empty(self.dimensions, dtype=np.uint8)

    def compute_interior(self):
        ny_local = self.dimensions[0]
        nx = self.dimensions[1]

        if ny_local <= 2:
            return

        for i in range(1, ny_local - 1):
            for j in range(nx):
                j_left = (j - 1 + nx) % nx
                j_right = (j + 1) % nx
                nb = (
                    self.cells[i - 1, j_left]  + self.cells[i - 1, j] + self.cells[i - 1, j_right] +
                    self.cells[i,     j_left]  +                         self.cells[i,     j_right] +
                    self.cells[i + 1, j_left]  + self.cells[i + 1, j] + self.cells[i + 1, j_right]
                )
                self._apply_rules(i, j, nb)

    def compute_borders(self, ghost_above, ghost_below):
        ny_local = self.dimensions[0]
        nx = self.dimensions[1]

        extended = np.vstack([ghost_above.reshape(1, nx), self.cells, ghost_below.reshape(1, nx)])

        border_rows = [0]
        if ny_local > 1:
            border_rows.append(ny_local - 1)

        for i in border_rows:
            ei = i + 1
            for j in range(nx):
                j_left = (j - 1 + nx) % nx
                j_right = (j + 1) % nx
                nb = (
                    extended[ei - 1, j_left]  + extended[ei - 1, j] + extended[ei - 1, j_right] +
                    extended[ei,     j_left]  +                       extended[ei,     j_right] +
                    extended[ei + 1, j_left]  + extended[ei + 1, j] + extended[ei + 1, j_right]
                )
                self._apply_rules(i, j, nb)

        self.cells = self._next_cells

class App:

    def __init__(self, geometry, grid):
        import pygame as pg
        self.pg = pg
        self.grid = grid
        # Taille d'une cellule en pixels
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
            
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.col_life = pg.Color("black")
        self.col_dead = pg.Color("white")
        self.screen = pg.display.set_mode((self.width, self.height))

    def compute_rectangle(self, i, j):
        return (self.size_x * j,self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def compute_color(self, i, j):
        if self.grid.cells[i, j] == 0:
            return self.col_dead
        else:
            return self.col_life

    def draw(self):
        pg = self.pg
        ny, nx = self.grid.dimensions
        for i in range(ny):
            for j in range(nx):
                self.screen.fill(self.compute_color(i, j), self.compute_rectangle(i, j))
        if self.draw_color is not None:
            for i in range(ny):
                pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y))
            for j in range(nx):
                pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height))
        pg.display.update()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        if rank == 0:
            print("Erreur : il faut au moins 2 processus.")
            print("  mpirun --oversubscribe -np P python game_of_life_mpi.py")
        sys.exit(1)

    dico_patterns = {
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }

    choice = 'glider'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])

    if rank == 0:
        try:
            init_pattern = dico_patterns[choice]
        except KeyError:
            print(f"Pattern inconnu '{choice}'. Disponibles : {list(dico_patterns.keys())}")
            comm.Abort(1)

        global_dim = init_pattern[0]
        ny_global = global_dim[0]
        nx_global = global_dim[1]
        global_grid = Grille(*init_pattern)
        print(f"Pattern initial : {choice}")
        print(f"Dimensions de la grille : {ny_global} x {nx_global}")
        print(f"Résolution écran : ({resx}, {resy})")
        print(f"Nombre de processus MPI : {size}")
        print(f"  - Rang 0 : affichage")
        print(f"  - Rang 1 : orchestrateur + calcul")
        if size > 2:
            print(f"  - Rangs 2..{size-1} : calcul")
    else:
        ny_global = None
        nx_global = None

    ny_global = comm.bcast(ny_global, root=0)
    nx_global = comm.bcast(nx_global, root=0)

    color = 0 if rank == 0 else 1
    compute_comm = comm.Split(color, rank)

    if rank != 0:
        compute_rank = compute_comm.Get_rank()
        compute_size = compute_comm.Get_size()

    if rank == 0:
        comm.Send(global_grid.cells, dest=1, tag=10)
    elif rank == 1:
        full_cells = np.empty((ny_global, nx_global), dtype=np.uint8)
        comm.Recv(full_cells, source=0, tag=10)

    if rank != 0:
        rows_per_proc = [ny_global // compute_size] * compute_size
        remainder = ny_global % compute_size
        for i in range(remainder):
            rows_per_proc[i] += 1

        offsets = [0]
        for i in range(compute_size - 1):
            offsets.append(offsets[-1] + rows_per_proc[i])

        ny_local = rows_per_proc[compute_rank]

        if compute_rank == 0:
            print(f"Répartition des lignes ({compute_size} processus de calcul) : {rows_per_proc}")

        if compute_rank == 0:
            sendbuf = full_cells.flatten()
            sendcounts = [rows_per_proc[i] * nx_global for i in range(compute_size)]
            displacements = [offsets[i] * nx_global for i in range(compute_size)]
        else:
            sendbuf = None
            sendcounts = None
            displacements = None

        local_cells = np.empty(ny_local * nx_global, dtype=np.uint8)
        compute_comm.Scatterv([sendbuf, sendcounts, displacements, MPI.UNSIGNED_CHAR], local_cells, root=0)
        local_cells = local_cells.reshape(ny_local, nx_global)

        local_grid = Grille.__new__(Grille)
        local_grid.dimensions = (ny_local, nx_global)
        local_grid.cells = local_cells
        
        compute_rank_above = (compute_rank - 1 + compute_size) % compute_size
        compute_rank_below = (compute_rank + 1) % compute_size
        
    if rank == 0:
        import pygame as pg
        pg.init()
        appli = App((resx, resy), global_grid)
        appli.draw()  
        
    mustContinue = True
    while mustContinue:
        
        if rank != 0:
            t1 = time.time()
            
            ghost_above = np.empty(nx_global, dtype=np.uint8)
            ghost_below = np.empty(nx_global, dtype=np.uint8)

            send_last = local_grid.cells[-1, :].copy()
            send_first = local_grid.cells[0, :].copy()

            req1 = compute_comm.Isend(send_last, dest=compute_rank_below, tag=0)
            req2 = compute_comm.Isend(send_first, dest=compute_rank_above, tag=1)
            req3 = compute_comm.Irecv(ghost_above, source=compute_rank_above, tag=0)
            req4 = compute_comm.Irecv(ghost_below, source=compute_rank_below, tag=1)
            
            local_grid.prepare_computation()
            local_grid.compute_interior()
            
            MPI.Request.Waitall([req1, req2, req3, req4])
            
            local_grid.compute_borders(ghost_above, ghost_below)

            t2 = time.time()
            
            local_flat = local_grid.cells.flatten()
            if compute_rank == 0:
                recvbuf = np.empty(ny_global * nx_global, dtype=np.uint8)
                recvcounts = [rows_per_proc[i] * nx_global for i in range(compute_size)]
                recv_displacements = [offsets[i] * nx_global for i in range(compute_size)]
            else:
                recvbuf = None
                recvcounts = None
                recv_displacements = None

            compute_comm.Gatherv( local_flat, [recvbuf, recvcounts, recv_displacements, MPI.UNSIGNED_CHAR], root=0)
            
            if compute_rank == 0:
                comm.Send(recvbuf, dest=0, tag=20)
                comm.send(t2 - t1, dest=0, tag=30)
                
        if rank == 0:
            assembled = np.empty(ny_global * nx_global, dtype=np.uint8)
            comm.Recv(assembled, source=1, tag=20)
            
            global_grid.cells = assembled.reshape(ny_global, nx_global)
            appli.grid = global_grid
            t_draw_start = time.time()
            appli.draw()
            t_draw_end = time.time()
            t_calcul = comm.recv(source=1, tag=30)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False

            print(f"Temps calcul : {t_calcul:2.2e}s | Temps affichage : {t_draw_end - t_draw_start:2.2e}s\r", end='')
            
        mustContinue = comm.bcast(mustContinue, root=0)
        
    if rank == 0:
        pg.quit()

    compute_comm.Free()
