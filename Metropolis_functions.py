import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # sudo pip3 install numba
import io
import imageio  # sudo pip3 install imageio

"""
This file contains functions for high speed Metropolis algorithm computation for the 2D XY-model using NumPy and Numba.
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
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnteic XY-model.
        
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
def Metropolis_slow_quench(J, L, relaxation_time, plot_at_Nth_index, lattice,
                           T_init, T_final, T_n, new_phi):
    """
    Applies the XY-model Metropolis evolution algorithm at temperature T_init to a given input lattice of size L×L, where each of the spins takes on value ranging from -π to π.
    After relaxation_time number of time-steps, the temperature of the system changes until we run for the final time for T_final. This is why the function is labelled slow quench.

    Parameters
    ----------
        J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnteic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.
        
        relaxation_time: int
            The number of Metropolis evolution time-steps one applies to the lattice for a given temperature T before changing the temperature.
        
        plot_at_Nth_index: np.ndarray (int)
            Specifies at which Metropolis evolution time-steps to save the lattice snapshot for plotting purposes.
        
        lattice: np.ndarray (float)
            The input lattice containing L×L spins.
        
        T_init: float
            The initial temperature of the system in units of kB (Boltzmann constant).
        
        T_final: float
            The final temperature of the system in units of kB.
        
        T_n: int
            The number of points between T_init and T_final, inclusive.
        
        new_phi: np.ndarray (float)
            A numpy array containing N uniformly generated numbers ranging from -π to π, where N = relaxation_time * T_n.
        
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
    N = relaxation_time * T_n
    T_array = np.linspace(T_init, T_final, T_n)  # In units of kB
    lattices_plot = np.zeros((len(plot_at_Nth_index), L, L))
    lattices_tau = np.zeros((T_n, L, L))
    T_history = np.zeros(len(plot_at_Nth_index))
    T_counter = 0
    for Nth_run in range(N):
        i, j = rand_2D_indices(L)
        dE = get_energy_difference_with_trial_state(J, L, i, j,
                                                    new_phi[Nth_run], lattice)
        if not (Nth_run == 0) and ((Nth_run % relaxation_time == 0) or
                                   (Nth_run == N - 1)):
            lattices_tau[T_counter, :, :] = lattice
            T_counter = T_counter + 1
        if (dE <= 0):
            lattice[i, j] = new_phi[Nth_run]
        else:
            r = np.random.rand()
            W = np.exp(
                -1 / T_array[T_counter] *
                dE) if T_array[T_counter] > 0 else 0  # T is in units of kB
            if (r < W):
                lattice[i, j] = new_phi[Nth_run]
        if Nth_run in plot_at_Nth_index:
            lattices_plot[np.where(
                plot_at_Nth_index == Nth_run)[0][0], :, :] = lattice
            T_history[np.where(
                plot_at_Nth_index == Nth_run)[0][0]] = T_array[T_counter]
    return lattices_tau, lattices_plot, T_history


def creategif(J, L, T, plot_at_Nth_index, lattices_plot, filename):
    """
    Creates an animated .gif of the evolution of the lattice in θ∈[-π,π) space. No temporary files are created as we are utilising RAM.

    Parameters
    ----------
        J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnteic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.
        
        T: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),). It stores the temperature of the system at time-steps specified by plot_at_Nth_index.
        
        lattices_plot: np.ndarray (float)
            A numpy array of shape (len(plot_at_Nth_index),L,L). It stores the snapshot of the lattice at time-steps specified by plot_at_Nth_index.
        
        filename: string
            Specifies the filename of the .gif to be saved.
    
    Returns
    -------
        None
    """
    with imageio.get_writer(filename, mode='I') as writer:
        for counter, i in enumerate(plot_at_Nth_index):
            print(
                f"Creating gif... {np.round((counter+1)/len(plot_at_Nth_index)*100,2)}%"
            )
            plt.imshow(lattices_plot[counter, :, :], cmap='hsv')
            plt.clim(-np.pi, np.pi)
            plt.colorbar(ticks=[-np.pi, 0, np.pi])
            plt.title(
                f"Run #{i}. $J$={J}. $T$={np.round(T[counter],2)}. $L$={L}.")
            plt.xlabel("$x$")
            plt.ylabel("$y$")
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