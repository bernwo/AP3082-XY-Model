import numpy as np
from Metropolis_functions import Metropolis_slow_quench, creategif
from Observables import get_energy_per_spin_per_lattice
import matplotlib.pyplot as plt
# Initialise all variables
J = 1
L = 64

tau = 10000000 # relaxation time a.k.a equilibrating time
nrelax = 1 # total number of times to let the system equilibrate
nframes = 201 # total number of frames in the .gif over the whole simulation
N = nrelax * tau # total number of iterations in the Metropolis algorithm
plot_at_Nth_index = np.linspace(0, N - 1, nframes).astype('int')

lattice = np.random.uniform(low=-np.pi, high=np.pi, size=(L,L))
# lattice = np.zeros((L, L))

Tc = 0.892937 * J  # http://www.lps.ens.fr/~krauth/images/7/72/Stage_Mayer_Johannes_2015.pdf in units of kB
T_init = 0
T_final = 0

lattices_relax, lattices_plot, T_history = Metropolis_slow_quench(
    J=J,
    L=L,
    relaxation_time=tau,
    plot_at_Nth_index=plot_at_Nth_index,
    lattice=lattice,
    T_init=T_init,
    T_final=T_final,
    T_n=nrelax)
creategif(J=J,
          L=L,
          T=T_history,
          plot_at_Nth_index=plot_at_Nth_index,
          lattices_plot=lattices_plot,
          filename="simulation_images/Metropolis_phase_energy.gif",
          plot_mode='phase_and_energy_spaces_arrows')

plt.close()
plt.imshow(lattices_relax[-1, :, :], cmap='hsv')
plt.clim(-np.pi, np.pi)
plt.colorbar(ticks=[-np.pi, 0, np.pi])
plt.title(f"Run #{tau*nrelax-1}. $J$={J}. $T/T_c={T_final}$. $L$={L}.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

X, Y = np.mgrid[0:L, 0:L]
U, V = np.cos(lattices_relax[- 1, :, :].T), np.sin(lattices_relax[- 1, :, :].T)
plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
plt.imshow(lattices_relax[- 1, :, :], cmap='hsv')
plt.clim(-np.pi, np.pi)
plt.colorbar(ticks=[-np.pi, 0, np.pi])
plt.title(f"Run #{tau*nrelax-1}. $J$={J}. $T/T_c={T_final}$. $L$={L}.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

E = get_energy_per_spin_per_lattice(J, lattices_relax[-1,:,:])
plt.close()
plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
plt.imshow(E, cmap='YlOrRd')
plt.clim(-4, 0)
plt.colorbar(ticks=[-4, -2, 0])
plt.title(f"Run #{tau*nrelax-1}. $J$={J}. $T/T_c={T_final}$. $L$={L}.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


fig = plt.figure(figsize=(10,3.7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

fig.suptitle(f"Run #{tau*nrelax-1}. $J$={J}. $T/T_c={T_final}$. $L$={L}.")
ax1.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
im1 = ax1.imshow(lattices_relax[- 1, :, :], vmin=-np.pi, vmax=np.pi, cmap='hsv')
fig.colorbar(im1,ticks=[-3.14, 0, 3.14], ax=ax1)
ax1.set_title("Phase space, $θ\in(-\pi,\pi)$")

ax2.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
im2 = ax2.imshow(E, vmin=-4, vmax=0, cmap='YlOrRd')
fig.colorbar(im2,ticks=[-4, -2, 0], ax=ax2)
ax2.set_title("Energy, $E$")