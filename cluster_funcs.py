import numpy as np
import matplotlib.pyplot as plt
from Observables_functions import get_energy_per_spin_per_lattice
from scipy.ndimage import convolve

def get_cluster_borders(array):
    """
    Get the positions of each cluster in array.
    
    Parameters:
    -----------
    array: nd.array
    
    Returns:
    -------
    nd.array: positions ordered by columns
    """
    addresses_clusters = []
    borders_clusters = []
    for i in range(1,np.max(array)+1):
        cluster = np.where(array == i)
        borders_clusters.append(get_borders(positions=cluster, L=len(array)))
    return borders_clusters

    

def get_borders(positions, L):
    """
    A cluster is represented as zeros and ones in a numpy array.
    This function finds the border of a cluster and returns the position
    of every element of the border.
    """
    #print(positions)
    if len(positions[0]) == 1:
        #print(positions)
        return np.array(positions).flatten()
    else:
        cluster = np.zeros((L,L))
        borders = np.zeros((L,L))
        cluster[positions] = 1
        kernel = np.array([[0, 1, 0],[1, 0, 1],[0, 1, 0]])
        neighbour_sums = convolve(cluster, kernel, mode='wrap')
        borders[np.where(neighbour_sums == 1)] = 1
        borders[np.where(neighbour_sums == 2)] = 1
        borders[np.where(neighbour_sums == 3)] = 1
        borders = borders*cluster
        border = np.where(borders != 0)
        #print('border: '+str(border))
        return np.array(border).T



def temp_prob(temperature, J=1):
    return 1 - np.exp(-J/np.abs(temperature))


def index_format(direction):
    """
    Position in matrix transformed for use in 
    """
    return np.split(direction, 2)


def get_random_cluster_element(addresses):
    """
    Returns a random element of a list addresses.
    """
    length = len(addresses)
    #print('addresses: '+str(addresses))
    if len(np.shape(addresses)) == 1:
        #print('normal element: '+str(addresses))
        return addresses
    
    else:
        element = addresses[np.random.choice(len(addresses))]
        #print('border element: '+str(element))
        return element


def get_random_neighbor(array=[[0, 1], [0, -1], [1, 0], [-1, 0]]):
    """
    Returns a random element from array.
    """
    return np.array(array[np.random.choice(range(len(array)))])


def propagate_xy(array, boolean_array, spins, random_vector, center, direction, temperature):

    initial = index_format(center)
    final = index_format((center+direction) % len(array))
    # check array not used before
    if boolean_array[final] and boolean_array[initial]:
        # check not previously part of the same cluster
        if not array[initial][0] == array[final][0]:
            # check spin dependent conditions
            if spins_aligned(element=center,
                             direction=direction,
                             spins=spins,
                             random_vector=random_vector):
                if temp_prob(temperature) > np.random.rand():
                    #print(temp_prob(temperature))
                    # add to cluster
                    array[np.where(array == array[final])] = array[initial]
                    # update boolean array
                    boolean_array[initial] = False
                    boolean_array[final] = False
    return array, boolean_array


def propagation_round_xy(array, spins, random_vector, temperature):
    """
    Propagate all the clusters in an array a single time according to the xy model.
    """
    #print('propagation round...')
    # Initialize boolean array at the beggining of the round
    boolean_array = array < np.max(array)+1
    cluster_elements = get_cluster_borders(array)
    #print('cluster elements: '+str(cluster_elements))
    #cluster_elements = np.delete(cluster_elements, np.where(len(cluster_elements) == 1))
    random_elements = [get_random_cluster_element(element) for element in cluster_elements][1::]
    #print(random_elements)
    for element in random_elements:
        direction = get_random_neighbor()

        array, boolean_array = propagate_xy(array=array,
                                         boolean_array=boolean_array,
                                         center=element,
                                         direction=direction,
                                           spins=spins,
                                           random_vector=random_vector,
                                           temperature=temperature)

    array = relabel_clusters(array)

    return array


def sweden_wang_cluster(spins, p, random_vector):

    L = len(spins)
    array = np.resize(np.arange(1, L*L+1), (L, L))

    cluster_elements = get_cluster_borders(array)

    random_elements = [get_random_cluster_element(element) for element in cluster_elements][1::]
    neighbors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    for element in random_elements:
        for neighbor in neighbors:

            initial = index_format(element)
            final = index_format((element+neighbor) % len(array))
            if spins_aligned(element, neighbor, spins, random_vector):
                if not array[initial][0] == array[final][0]:
                    if np.random.rand() < p:
                        array[np.where(array == array[final])] = array[initial]

    return array


def sweden_wang_evolution(spins, temperature, J=1):

    random_vector = get_vector_components(2*np.pi*np.random.rand())
    p = 1 - np.exp(-J/temperature)
    clusters = sweden_wang_cluster(spins, p, random_vector)
    spins = rotate_percolated_cluster(spins, clusters, random_vector)

    return spins

def propagation_round(array, p):
    """
    Propagate all the clusters in an array a single time according to a percolation model with probability p.
    """
    boolean_array = array < np.max(array)+1
    cluster_elements = get_cluster_borders(array)
    #print('cluster elements: '+str(cluster_elements))
    #cluster_elements = np.delete(cluster_elements, np.where(len(cluster_elements) == 1))
    random_elements = [get_random_cluster_element(element) for element in cluster_elements][1::]
    #print('random elements: '+str(random_elements))

    for element in random_elements:
        direction = get_random_neighbor()

        if np.random.rand() < p:
            array, boolean_array = propagate(array=array,
                                             boolean_array=boolean_array,
                                             center=element,
                                             direction=direction)
    array = relabel_clusters(array)

    return array


def propagate(array, boolean_array, center, direction):
    """
    Execute round of propagation in array for a single cluster.
    
    Parameters:
    ----------
    array: nd.darray
    boolean_array: nd.darray
    center: nd.array
    direction: n.array

    Returns:
    -------
    array: nd.array
    boolean_array: 
    """
    #center = np.array(center).flatten()
    #print('center: '+str(center))
    initial = index_format(center)
    final = index_format((center+direction) % len(array))
    
    if boolean_array[final] and boolean_array[initial]:
        if not array[initial][0] == array[final][0]:
            array[np.where(array == array[final])] = array[initial]
            boolean_array[initial] = False
            boolean_array[final] = False
            #array = relabel_clusters(array)
    return array, boolean_array


def spins_aligned(element, direction, spins, random_vector):
    """
    Check if two spins are aligned.
    Parameters:
    ----------
    element: nd.array
    direction: nd.array
    spins: nd.darray
    random_vector: nd.array
    
    Returns:
    -------
    boolean
    """
    initial = index_format(element)
    final = index_format((element+direction) % len(spins))
    #print(get_vector_components(spins[initial]).flatt)
    #print(random_vector)

    proj1 = np.dot(get_vector_components(spins[initial]).flatten(), random_vector)
    proj2 = np.dot(get_vector_components(spins[final]).flatten(), random_vector)
    

    if proj1*proj2 > 0:
        return True
    else:
        return False


def get_vector_components(angle):
    """Get x and y components of a unit vector with angle"""
    return np.array([np.cos(angle), np.sin(angle)])


def percolation(invaded):
    """
    Check if percolation happened in invaded. That is, check if there is a cluster
    spanning the lattice along a given direction.

    Parameters:
    ----------
    invaded: nd.array

    Returns:
    -------
    boolean
    """

    L = len(invaded)
    ones_vec = np.ones(L)
    ones_cluster = np.zeros((L, L))
    ones_cluster[np.where(invaded == np.max(invaded))] = 1
    path_1 = ones_cluster.dot(ones_vec)
    path_2 = ones_cluster.T.dot(ones_vec)

    if 0. not in path_1 or 0. not in path_2:
        percolation = True
    else:
        percolation = False

    return percolation


def relabel_clusters(array):
    """
    At the end of a propagation round, clusters label reorders the cluster from
    1 to the total number of clusters than can be found by the number of different
    elements in array.
    
    Parameters:
    ----------
    array: nd.array
    
    Returns:
    -------
    array: nd.array
    """
    aux = np.copy(array)
    bins = np.bincount(aux.flatten())
    dat = np.nonzero(bins)[0]
    i = 1
    sorted_dat = np.array(sorted(zip(bins[bins != 0], dat)))

    for element in sorted_dat:
        if element[0] != 0:
            array[np.where(aux == element[1])] = i
            i += 1
    return array


def xy_model(L, n_steps=1):
    """
    Executes IC algorithm for xy model of size L x L during 
    n_step for temperature.
    """
    spins = 2*np.pi*np.random.rand(L, L)
    invaded = np.resize(np.arange(1, L*L+1), (L, L))
    temperature = 1E-10
    ts = []
    ims = []
    n = 0
    perc = False
    while n < n_steps:

        while not perc:

            random_vector = get_vector_components(2*np.pi*np.random.rand())
            invaded = propagation_round_xy(invaded, spins, random_vector, temperature)
            perc = percolation(invaded)

        spins = rotate_percolated_cluster(spins, invaded, random_vector)
        im = plt.imshow(spins, animated=True)
        ims.append([im])
        ts.append(temperature)
        temperature = get_temperature(spins)
        n += 1

    return spins, invaded, ts, ims


def rotate_percolated_cluster(spins, invaded, random_vector):
    """
    Rotate spins for each cluster from invaded around random vector
    with probability 1/2.
    """
    random_angle = np.arctan(random_vector[1]/random_vector[0])
    for i in range(np.max(invaded.flatten())):
        if np.random.rand() < 1/2:
            spins[np.where(invaded == i+1)] += random_angle
    spins = spins % (2*np.pi)
    return spins


def get_temperature(spins):
    """Use equipartition theorem to calculate kbT"""
    return -(2/3)*np.sum(np.sum(get_energy_per_spin_per_lattice(J=1, lattice=spins)))


def percolation_model(L, p):
    """
    Execute percolation model with probability p on a lattice of size L x L.

    Parameters:
    ----------
    L: int
    P:float

    Returns:
    -------
    ims: nd.darray of plt.axes objects
    invaded: np.darray
    """
    invaded = np.resize(np.arange(1, L*L+1), (L,L))
    ims = [[plt.imshow(invaded, animated=True)]]
    #step = 1
    perc = False
    while not perc:
        invaded = propagation_round(invaded, p)
        im = plt.imshow(invaded, animated=True)
        ims.append([im])
        perc = percolation(invaded)
        # print(percolation(invaded), invaded)
    return ims, invaded