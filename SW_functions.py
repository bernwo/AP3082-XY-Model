import numpy as np

"""
    Functions used to implement the Swedensen-Wang algorithm.
"""

def rotate_clusters(spins, invaded, random_vector):
    """
    Rotate spins for each cluster from invaded around random vector
    with probability 1/2.
    """
    x, y = random_vector
    rot_mat = np.array([[x,-y],[y,x]])
    #print(rot_mat)
    for i in range(int(np.max(invaded.flatten()))):
        #print(invaded)
        if np.random.rand() < 1/2:
            for element in np.column_stack(np.where(invaded == i+1)):
                #print(spins)
                #print(spins[tuple(element)] - 2* np.dot(spins[tuple(element)],random_vector)* spins[tuple(element)])
                spins[tuple(element)] = spins[tuple(element)] - 2* np.dot(spins[tuple(element)],random_vector)* random_vector

                #print(spins)

    return spins


def check_neighbours(spins,cluster,seed,T,random_vector,cluster_number,J=1):
    """
    Recursive propagation of a single cluster
    """
    neighbors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    for neighbor in neighbors:

        goto = (seed + neighbor)%len(spins)
        #print('goto')
        #print(goto)
        projection = np.dot(random_vector,spins[tuple(goto)])*np.dot(random_vector,spins[tuple(seed)])

        cluster[tuple(seed)] = cluster_number
        #print(projection)
        if projection>0 and cluster[tuple(goto)] == 0:

            p =  1-np.exp(-2*projection/T)
            #print(p)
            #print(percolation(cluster))
            if np.random.rand() < p:

                check_neighbours(spins,cluster,goto,T,random_vector,cluster_number)
    return cluster


def sw_evolution(spins, T, J=1):
    """
    Evolve clusters covering all spin connected regions.

    Parameters:
    -----------
    spins: nd.ndarray
    T: float

    Returns:
    spins: nd.ndarray
    clusters: nd.ndarray
    """
    L = len(spins)
    #print(spins)
    clusters = np.zeros((L,L))
    random_vector = get_vector_components(2*np.pi*np.random.rand())
    cluster_number = 1
    seed = np.random.randint(low=0,high=L-1,size=2)
    rest = np.column_stack(np.where(clusters==0))
    while len(rest) > 0:
        seed = rest[np.random.choice(len(rest))]
        #print('seed')
        #print(seed)
        clusters = check_neighbours(spins,clusters,seed,T,random_vector,cluster_number)
        #print(clusters)
        cluster_number += 1
        rest = np.column_stack(np.where(clusters==0))

    spins = rotate_clusters(spins, clusters, random_vector)
    #print(spins)
    return spins, clusters

def get_vector_components(angle):
    """Get x and y components of a unit vector with angle"""
    return np.array([np.cos(angle), np.sin(angle)])
