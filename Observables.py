import numpy as np
from numba import jit


@jit(nopython=True)
def get_squared_magnetisation(L, T_n, lattices_relax):
    M2 = np.zeros(T_n)
    for i in range(T_n):
        M2[i] = np.sum(np.cos(lattices_relax[i,:,:]))**2 \
                 + np.sum(np.sin(lattices_relax[i,:,:]))**2
    M2 = M2 / np.power(L, 4)
    return M2

def get_energy(J, T_n, lattices_relax):
    E = np.zeros(T_n)
    for i in range(T_n):
        E[i] = np.mean(-J *
                       (np.cos(lattices_relax[i, :, :] -
                               np.roll(lattices_relax[i, :, :], 1, axis=0)) +
                        np.cos(lattices_relax[i, :, :] -
                               np.roll(lattices_relax[i, :, :], -1, axis=0)) +
                        np.cos(lattices_relax[i, :, :] -
                               np.roll(lattices_relax[i, :, :], 1, axis=1)) +
                        np.cos(lattices_relax[i, :, :] -
                               np.roll(lattices_relax[i, :, :], -1, axis=1))))
    return E

def get_squared_energy(J, T_n, lattices_relax):
    E2 = np.zeros(T_n)
    for i in range(T_n):
        E2[i] = np.mean(
            (-J * (np.cos(lattices_relax[i, :, :] -
                          np.roll(lattices_relax[i, :, :], 1, axis=0)) +
                   np.cos(lattices_relax[i, :, :] -
                          np.roll(lattices_relax[i, :, :], -1, axis=0)) +
                   np.cos(lattices_relax[i, :, :] -
                          np.roll(lattices_relax[i, :, :], 1, axis=1)) +
                   np.cos(lattices_relax[i, :, :] -
                          np.roll(lattices_relax[i, :, :], -1, axis=1))))**2)
    return E2

def get_specificheat(J, T_n, lattices_relax, T):
       E2 = get_squared_energy(J, T_n, lattices_relax)
       E = get_energy(J, T_n, lattices_relax)
       c = (E2-E**2)/((T**2))
       return c