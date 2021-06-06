import numpy as np
import matplotlib.pyplot as plt

def get_clusters(array):
    addresses_clusters = []
    for i in range(1, np.max(array)+1):
        addresses_clusters.append(np.transpose(np.where(array == i)))
    return addresses_clusters


def index_format(direction):
    return np.split(direction, 2)


def get_random_cluster_element(addresses):
    length = len(addresses)
    return addresses[np.random.choice(range(length))]


def get_random_neighbor(array=[[0, 1], [0, -1], [1, 0], [-1, 0]]):
    return np.array(array[np.random.choice(range(len(array)))])


def propagate(array, boolean_array, center, direction):

    initial = index_format(center)
    final = index_format((center+direction) % len(array))

    if boolean_array[final] and boolean_array[initial]:
        if not array[initial][0] == array[final][0]:
            array[np.where(array == array[final])] = array[initial]
            boolean_array[initial] = False
            boolean_array[final] = False
            #array = relabel_clusters(array)
    return array, boolean_array


def propagation_round_xy(array, spins, random_vector):

    boolean_array = array < np.max(array)+1
    cluster_addresses = get_clusters(array)
    random_elements = [get_random_cluster_element(cluster_addresses[i]) for i in range(len(cluster_addresses))]

    for element in random_elements:
        direction = get_random_neighbor()

        if spins_aligned(element, direction, spins):
            array, boolean_array = propagate(array=array,
                                             boolean_array=boolean_array,
                                             center=element,
                                             direction=direction)

    return array


def propagation_round(array, p):

    boolean_array = array < np.max(array)+1
    cluster_addresses = get_clusters(array)
    random_elements = [get_random_cluster_element(cluster_addresses[i]) for i in range(len(cluster_addresses))]

    for element in random_elements:
        direction = get_random_neighbor()

        if np.random.rand() < p:
            array, boolean_array = propagate(array=array,
                                             boolean_array=boolean_array,
                                             center=element,
                                             direction=direction)
    array = relabel_clusters(array)

    return array


def spins_aligned(element, direction, spins, random_vector):
    initial = index_format(element)
    final = index_format((element+direction) % len(array))
    return np.dot(spins[initial], random_vector) * np.dot(spins[final], random_vector) > 0


def get_vector_components(angle):
    return [np.cos(angle), np.sin(angle)]


def percolation(invaded):

    L = len(invaded)
    #invaded = relabel_clusters(invaded)
    ones_vec = np.ones(L)
    ones_cluster = np.zeros((L, L))
    ones_cluster[np.where(invaded == np.max(invaded))] = 1
    path_1 = ones_cluster.dot(ones_vec)
    path_2 = ones_cluster.T.dot(ones_vec)
    #print(invaded)
    #print(ones_cluster)
    #print(path_1, path_2)

    if 0. not in path_1 or 0. not in path_2:
        percolation = True
    else:
        percolation = False

    return percolation


def relabel_clusters(array):
    aux = np.copy(array)
    bins = np.bincount(aux.flatten())
    dat = np.nonzero(bins)[0]
    #print(bins)
    #print(dat)
    i = 1
    sorted_dat = np.array(sorted(zip(bins[bins != 0], dat)))

    for element in sorted_dat:
        if element[0] != 0:
            array[np.where(aux == element[1])] = i
            i += 1
    return array