import matplotlib.pyplot as plt

# === Données mesurées ===
sizes = [1000, 10000, 50000]
proc_counts = [1, 2, 4, 8]

# Temps séquentiel
t_seq = {1000: 0.007140, 10000: 0.580584, 50000: 14.831156}

# Temps MPI naïf (bcast + filtre)
t_naive = {
    2: {1000: 0.016835, 10000: 1.533654, 50000: 39.356118},
    4: {1000: 0.005415, 10000: 0.454317, 50000: 11.292708},
    8: {1000: 0.007943, 10000: 0.096551, 50000: 2.195280},
}

# Temps MPI optimisé avec sendrecv
t_sendrecv = {
    2: {1000: 0.015935, 10000: 1.548298, 50000: 40.831857},
    4: {1000: 0.004945, 10000: 0.437385, 50000: 11.163191},
    8: {1000: 0.002355, 10000: 0.083550, 50000: 2.060051},
}

# Temps MPI optimisé avec alltoall
t_alltoall = {
    2: {1000: 0.016645, 10000: 1.523203, 50000: 38.967862},
    4: {1000: 0.004978, 10000: 0.413478, 50000: 11.076285},
    8: {1000: 0.001461, 10000: 0.080447, 50000: 2.040665},
}

all_versions = [
    ("Naïf", t_naive),
    ("Sendrecv", t_sendrecv),
    ("Alltoall", t_alltoall),
]

print("Speedup")
print(f"{'Version':<25} {'N=1000':>10} {'N=10000':>10} {'N=50000':>10}")
print("-" * 55)
for label, data in all_versions:
    for p in [2, 4, 8]:
        vals = [t_seq[s] / data[p][s] for s in sizes]
        print(f"{label+' ('+str(p)+' proc)':<25} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}")

print("\nEfficacité")
print(f"{'Version':<25} {'N=1000':>10} {'N=10000':>10} {'N=50000':>10}")
print("-" * 55)
for label, data in all_versions:
    for p in [2, 4, 8]:
        vals = [t_seq[s] / (p * data[p][s]) for s in sizes]
        print(f"{label+' ('+str(p)+' proc)':<25} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}")

# === Courbes ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
procs = [1, 2, 4, 8]

styles = {
    "Naïf": ('o', '--'),
    "Sendrecv": ('s', '-.'),
    "Alltoall": ('^', '-'),
}

# --- Speedup ---
ax1 = axes[0]
ax1.plot(1, 1, 'kD', markersize=8, label="Séquentiel", zorder=5)

for size in sizes:
    for label, data in all_versions:
        marker, line = styles[label]
        sp = [1.0] + [t_seq[size] / data[p][size] for p in [2, 4, 8]]
        ax1.plot(procs, sp, marker=marker, linestyle=line, label=f"{label} N={size}")

ax1.plot(procs, procs, 'k:', linewidth=1.5, label="Idéal")
ax1.set_xlabel("Nombre de processus (P)", fontsize=12)
ax1.set_ylabel("Speedup S(P)", fontsize=12)
ax1.set_title("Speedup", fontsize=14)
ax1.legend(fontsize=6, loc="upper left", ncol=2)
ax1.set_xticks(procs)
ax1.grid(True, alpha=0.3)

# --- Efficacité ---
ax2 = axes[1]
ax2.plot(1, 1, 'kD', markersize=8, label="Séquentiel", zorder=5)

for size in sizes:
    for label, data in all_versions:
        marker, line = styles[label]
        eff = [1.0] + [t_seq[size] / (p * data[p][size]) for p in [2, 4, 8]]
        ax2.plot(procs, eff, marker=marker, linestyle=line, label=f"{label} N={size}")

ax2.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, label="Idéal")
ax2.set_xlabel("Nombre de processus (P)", fontsize=12)
ax2.set_ylabel("Efficacité E(P)", fontsize=12)
ax2.set_title("Efficacité", fontsize=14)
ax2.legend(fontsize=6, loc="best", ncol=2)
ax2.set_xticks(procs)
ax2.set_ylim(0, 1.15)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/speedup.png", dpi=150, bbox_inches='tight')
plt.savefig("/efficacite.png", dpi=150, bbox_inches='tight')