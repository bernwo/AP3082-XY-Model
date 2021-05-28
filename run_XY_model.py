import numpy as np
from Metropolis_functions import Metropolis_slow_quench, creategif
import matplotlib.pyplot as plt
# Initialise all variables
J = 1
L = 256

tau = 10000000 # relaxation time a.k.a equilibrating time
nrelax = 25 # total number of times to let the system equilibrate
nframes = 301 # total number of frames in the .gif over the whole simulation
N = nrelax * tau # total number of iterations in the Metropolis algorithm
plot_at_Nth_index = np.linspace(0, N - 1, nframes).astype('int')

# lattice = np.random.uniform(low=-np.pi, high=np.pi, size=(L,L))
lattice = np.zeros((L, L))

Tc = 0.892937 * J  # http://www.lps.ens.fr/~krauth/images/7/72/Stage_Mayer_Johannes_2015.pdf in units of kB

new_phi = np.random.uniform(low=-np.pi, high=np.pi, size=N)
lattices_relax, lattices_plot, T_history = Metropolis_slow_quench(
    J=J,
    L=L,
    relaxation_time=tau,
    plot_at_Nth_index=plot_at_Nth_index,
    lattice=lattice,
    T_init=0,
    T_final=2 * Tc,
    T_n=nrelax,
    new_phi=new_phi)
creategif(J=J,
          L=L,
          T=T_history,
          plot_at_Nth_index=plot_at_Nth_index,
          lattices_plot=lattices_plot,
          filename="simulation_images/Metropolis.gif")

plt.close()
plt.imshow(lattices_plot[len(plot_at_Nth_index) - 1, :, :], cmap='hsv')
plt.clim(-np.pi, np.pi)
plt.colorbar(ticks=[-np.pi, 0, np.pi])
plt.title(f"Run #{plot_at_Nth_index[-1]}. $J$={J}. $L$={L}.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()