import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # sudo pip3 install numba
import io
import imageio  # sudo pip3 install imageio
from Observables import get_energy_per_spin_per_lattice, get_energy, get_magnetisation_squared
"""
This file contains functions for Metropolis algorithm computation for the 2D XY-model using NumPy and Numba.
"""

@jit(nopython=True)
def rand_2D_indices(L):
    """
    Generates row index and column index ranging from 0 to L-1 in a uniformly random manner.

    Parameters
    ----------
        L: int
            Lattice size where the total number of spins is given by L×L.
    
    Returns
    -------
        i: int
            Uniformly generated random number ranging from 0 to L-1. It is used to index the lattice.

        j: int
            Uniformly generated random number ranging from 0 to L-1. It is used to index the lattice.
    """
    i, j = np.random.randint(low=0, high=L, size=2)
    return i, j


@jit(nopython=True)
def get_energy_difference_with_trial_state(J, L, i, j, new_phi, lattice):
    """
    Get the energy difference between the trial state and the current state.

    Parameters
    ----------
        J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnetic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.

        i: int
            Uniformly generated random number ranging from 0 to L-1. It is used to index the lattice.

        j: int
            Uniformly generated random number ranging from 0 to L-1. It is used to index the lattice.

        new_phi: np.ndarray (float)
            A numpy array containing N uniformly generated numbers ranging from -π to π, where N = relaxation_time * T_n.
        
        lattice: np.ndarray (float)
            The input lattice containing L×L spins.
        
    Returns
    -------
        dE: float
            Energy difference between the new and current lattice such that dE = E_new - E_current.
    """
    old = np.cos( lattice[i , j] - lattice[(i+1)%L , j] ) \
        + np.cos( lattice[i , j] - lattice[(i-1)%L , j] ) \
        + np.cos( lattice[i , j] - lattice[i , (j+1)%L] ) \
        + np.cos( lattice[i , j] - lattice[i , (j-1)%L] )

    new = np.cos( new_phi - lattice[(i+1)%L , j] ) \
        + np.cos( new_phi - lattice[(i-1)%L , j] ) \
        + np.cos( new_phi - lattice[i , (j+1)%L] ) \
        + np.cos( new_phi - lattice[i , (j-1)%L] )

    dE = -J * (new - old)
    return dE

@jit(nopython=True)
def Metropolis_single_iteration(J, L, lattice, T):
    i, j = rand_2D_indices(L)
    new_phi = (2 * np.pi) * np.random.rand() - np.pi
    dE = get_energy_difference_with_trial_state(J, L, i, j, new_phi, lattice)
    if (dE <= 0):
        lattice[i, j] = new_phi
    else:
        r = np.random.rand()
        W = np.exp(-1 / T * dE)  # T is in units of kB
        if (r < W):
            lattice[i, j] = new_phi

@jit(nopython=True)
def Metropolis(J, L, relaxation_time, extra_time, lattice,
                           T_init, T_final, T_n, plot_at_Nth_index, save_for_plot=False):
    """
    Applies the XY-model Metropolis evolution algorithm at temperature T_init to a given input lattice of size L×L, where each of the spins takes on value ranging from -π to π.
    After relaxation_time number of time-steps, the temperature of the system changes until we run for the final time for T_final. This is why the function is labelled slow quench.

    Parameters
    ----------
        J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnetic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.
        
        relaxation_time: int
            The number of Metropolis evolution time-steps one applies to the lattice for a given temperature T before changing the temperature.
        
        lattice: np.ndarray (float)
            The input lattice containing L×L spins.
        
        T_init: float
            The initial temperature of the system in units of kB (Boltzmann constant).
        
        T_final: float
            The final temperature of the system in units of kB.
        
        T_n: int
            The number of points between T_init and T_final, inclusive.
        
        plot_at_Nth_index: np.ndarray (int)
            Specifies at which Metropolis evolution time-steps to save the lattice snapshot for plotting purposes.
        
    Returns
    -------
        lattices_tau: np.ndarray (float)
            A numpy array of shape (T_n,L,L). It stores the snapshot of the lattice after relaxation ends for each temperature points, specified by T_array.
            This is used for studying physical observables of the system.

        lattices_plot: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),L,L). It stores the snapshot of the lattice at time-steps specified by plot_at_Nth_index.
            This is used for plotting the evolution of the lattice.

        T_history: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),). It stores the temperature of the system at time-steps specified by plot_at_Nth_index.
            This is used for plotting the evolution of the lattice.
    """
    T_array = np.linspace(T_init, T_final, T_n)  # In units of kB
    ave_M2 = np.zeros(T_n)
    ave_E = np.zeros(T_n)
    ave_E2 = np.zeros(T_n)
    total_counter = 0

    if save_for_plot:
        lattices_plot = np.zeros((len(plot_at_Nth_index), L, L))
        T_history = np.zeros(len(plot_at_Nth_index))

    for a in range(T_n):
        for b in range(relaxation_time):
            Metropolis_single_iteration(J, L, lattice, T_array[a])
        for c in range(extra_time):
            Metropolis_single_iteration(J, L, lattice, T_array[a])
            E = get_energy(J, L,lattice)
            ave_E2[a] += E**2
            ave_E[a] += E
            ave_M2[a] += get_magnetisation_squared(lattice)

        if save_for_plot and (total_counter in plot_at_Nth_index):
            lattices_plot[np.where(
                plot_at_Nth_index == total_counter)[0][0], :, :] = lattice
            T_history[np.where(
                plot_at_Nth_index == total_counter)[0][0]] = T_array[a]
        total_counter += 1

    ave_M2 = ave_M2/(extra_time*L**4)
    ave_E2 = ave_E2/(extra_time*4)
    ave_E = (ave_E/(extra_time*2))**2
    Cv = (ave_E2-ave_E)
    return ave_M2, Cv, lattices_plot, T_history


def creategif(J, L, T, plot_at_Nth_index, lattices_plot, filename, plot_mode):
    """
    Creates an animated .gif of the evolution of the lattice in θ∈[-π,π) space. No temporary files are created as we are utilising RAM.

    Parameters
    ----------
        J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnetic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.
        
        T: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),). It stores the temperature of the system at time-steps specified by plot_at_Nth_index.
        
        lattices_plot: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),L,L). It stores the snapshot of the lattice at time-steps specified by plot_at_Nth_index.
        
        filename: string
            Specifies the filename of the .gif to be saved.

        plot_mode: string
            Specifies the type of animated .gif plot to be produced. The possible options are:
            1.) phase_space_noarrows
            2.) phase_space_arrows
            3.) energy_space_arrows
            4.) phase_and_energy_spaces_arrows

    Returns
    -------
        None
    """
    assert (plot_mode == 'phase_space_noarrows') or (
        plot_mode
        == 'phase_space_arrows') or (plot_mode == 'energy_space_arrows') or (
            plot_mode == 'phase_and_energy_spaces_arrows')
    X, Y = np.mgrid[0:L, 0:L]
    with imageio.get_writer(filename, mode='I') as writer:
        for counter, i in enumerate(plot_at_Nth_index):
            print(
                f"Creating gif... {np.round((counter+1)/len(plot_at_Nth_index)*100,2)}%"
            )
            if plot_mode == 'phase_space_noarrows':
                plt.imshow(lattices_plot[counter, :, :], cmap='hsv')
                plt.clim(-np.pi, np.pi)
                plt.colorbar(ticks=[-np.pi, 0, np.pi])
            elif plot_mode == 'phase_space_arrows':
                U, V = np.cos(lattices_plot[counter, :, :].T), np.sin(
                    lattices_plot[counter, :, :].T)
                plt.quiver(X,
                           Y,
                           U,
                           V,
                           edgecolor='k',
                           facecolor='None',
                           linewidth=.5)
                plt.imshow(lattices_plot[counter, :, :], cmap='hsv')
                plt.clim(-np.pi, np.pi)
                plt.colorbar(ticks=[-np.pi, 0, np.pi])
            elif plot_mode == 'energy_space_arrows':
                U, V = np.cos(lattices_plot[counter, :, :].T), np.sin(
                    lattices_plot[counter, :, :].T)
                E = get_energy_per_spin_per_lattice(
                    J, lattices_plot[counter, :, :])
                plt.quiver(X,
                           Y,
                           U,
                           V,
                           edgecolor='k',
                           facecolor='None',
                           linewidth=.5)
                plt.imshow(E, cmap='YlOrRd')
                plt.clim(-4, 0)
                plt.colorbar(ticks=[-4, -2, 0])
            elif plot_mode == 'phase_and_energy_spaces_arrows':
                U, V = np.cos(lattices_plot[counter, :, :].T), np.sin(
                    lattices_plot[counter, :, :].T)
                E = get_energy_per_spin_per_lattice(
                    J, lattices_plot[counter, :, :])
                fig = plt.figure(figsize=(10, 3.7))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                ax1.quiver(X,
                           Y,
                           U,
                           V,
                           edgecolor='k',
                           facecolor='None',
                           linewidth=.5)
                im1 = ax1.imshow(lattices_plot[counter, :, :],
                                 vmin=-np.pi,
                                 vmax=np.pi,
                                 cmap='hsv')
                fig.colorbar(im1, ticks=[-3.14, 0, 3.14], ax=ax1)
                ax1.set_title("Phase space, $θ\in(-\pi,\pi)$")

                ax2.quiver(X,
                           Y,
                           U,
                           V,
                           edgecolor='k',
                           facecolor='None',
                           linewidth=.5)
                im2 = ax2.imshow(E, vmin=-4, vmax=0, cmap='YlOrRd')
                fig.colorbar(im2, ticks=[-4, -2, 0], ax=ax2)
                ax2.set_title("Energy, $E$")
            if plot_mode != 'phase_and_energy_spaces_arrows':
                plt.title(
                    f"Run #{i}. $J$={J}. $T$={np.round(T[counter],2)}. $L$={L}."
                )
            else:
                fig.suptitle(
                    f"Run #{i}. $J$={J}. $T$={np.round(T[counter],2)}. $L$={L}."
                )
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            image = imageio.imread(buf)
            writer.append_data(image)
            buf.close()
            plt.cla()
            plt.clf()
            plt.close('all')
    print(f"Gif created.")