import numpy as np
from numba import jit

@jit(nopython=True)
def get_magnetisation_squared(lattice):
    """
    Calculates and returns the magnetisation squared sum over the lattice.

    Parameters
    ----------
        lattice: np.ndarray (float)
            The input lattice containing L×L spins.
    
    Returns
    -------
        M2: float
            Magnetisation squared sum over the lattice.
    """
    M2 = np.sum(np.cos(lattice))**2 + np.sum(np.sin(lattice))**2
    return M2

@jit(nopython=True)
def get_energy(J, L, lattice):
	"""
	Calculates and returns the energy sum over the lattice.

    Parameters
    ----------
		J: float
            Coupling constant. It must follow that J>0 such that we are studying the ferromagnetic XY-model.
        
        L: int
            Lattice size where the total number of spins is given by L×L.
		
        lattice: np.ndarray (float)
            The input lattice containing L×L spins.
    
    Returns
    -------
        E: float
            Energy sum over the lattice.
	"""
	a = np.arange(0, L, 1, dtype=np.int32)

	e_right = np.cos(lattice - lattice[:, a - 1])
	e_left = np.cos(lattice - lattice[:, a - (L - 1)])
	e_down = np.cos(lattice - lattice[a - 1, :])
	e_up = np.cos(lattice - lattice[a - (L - 1), :])

	e = -J * (e_right + e_left + e_down + e_up)
	E = np.sum(e)
	return E

def get_energy_per_spin_per_lattice(J, lattice):
	"""
	Calculates and returns the energy sum over the lattice.

	Parameters
	----------
		J: float
			Coupling constant. It must follow that J>0 such that we are studying the ferromagnetic XY-model.
		
		lattice: np.ndarray (float)
			The input lattice containing L×L spins.

	Returns
	-------
		E: np.ndarray (float)
			Energy of each spin in the lattice.
	"""
	E = -J * (np.cos(lattice - np.roll(lattice, 1, axis=0)) +
				np.cos(lattice - np.roll(lattice, -1, axis=0)) +
				np.cos(lattice - np.roll(lattice, 1, axis=1)) +
				np.cos(lattice - np.roll(lattice, -1, axis=1)))
	return E