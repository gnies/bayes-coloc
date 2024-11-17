import numpy as np

dtype = [('i', 'i4'), 
    ('j', 'i4'), 
    ('flow', 'i4'),
    ('log_prob_swap_with_0', 'f8'), 
    ('log_prob_swap_with_1', 'f8'), 
    ('log_prob_swap_with_2', 'f8'), 
    ('log_prob_swap_with_3', 'f8'), 
    ('log_prob_swap_total', 'f8'),
    ('edge_type', 'i4'),
    ('used', 'bool')
]

class Graph:
    """ This is the data structure I came up with to represent the assignment graph.
        Each edge has the following attributes:
        - i: the first node
        - j: the second node
        - flow: the flow of the edge
        - log_prob_swap_with_0: the log probability of swapping the edge with an assigned pair
        - log_prob_swap_with_1: the log probability of swapping the edge with an unassigned pair in the first set
        - log_prob_swap_with_2: the log probability of swapping the edge with an unassigned pair in the second set
        - log_prob_swap_with_3: the log probability of swapping the edge with a bin-bin pair
        - log_prob_swap_total: the log probability of swapping the edge with any pair
        - type: the type of the edge (0: assigned, 1: unassinged first set, 2: unassigned second set, 3: bin_bin)
        - used: whether the edge is used
    """
    def __init__(self, n, m):
        max_edge_number = n + m
        self.n = n
        self.m = m
        self.memory = np.zeros(max_edge_number, dtype=dtype)
        self.memory['used'] = False
        self.memory['i'] = n
        self.memory['j'] = m

    def __str__(self):
        return str(self.memory[self.memory['used']])

    def __repr__(self):
        return str(self.memory[self.memory['used']])

    def numpy_graph(self):
        # return a sliced view of the memory array
        last_edge_index = np.where(self.memory['used'] == True)[0][-1]
        return self.memory[:last_edge_index+1]

    def add_edge(self, i, j,
            flow,
            log_p_swap_with_0,
            log_p_swap_with_3,
            log_p_swap_with_1,
            log_p_swap_with_2,
            log_p_swap_total,
            edge_type):
        """ Add an edge to the graph. """
        # find an empty slot in memory
        index = np.where(self.memory['used'] == False)[0][0]
        self.memory[index] = (i,j,
            flow,
            log_p_swap_with_0,
            log_p_swap_with_3,
            log_p_swap_with_1,
            log_p_swap_with_2,
            log_p_swap_total,
            edge_type, 
            True)

    def delete_edge(self, i, j):
        """ Delete an edge from the graph. """
        a = np.equal(i, self.memory['i'])
        b = np.equal(j, self.memory['j'])
        c = self.memory['used']
        cond = np.logical_and(a, b)
        cond = np.logical_and(cond, c)
        index = np.where(cond)[0][0]
        # on deletion we always free the memory slot of the last edge in the array
        # find last edge index
        last_edge_index = np.where(self.memory['used'] == True)[0][-1]
        # copy last edge to the deleted edge
        self.memory[index] = self.memory[last_edge_index]
        # free last edge memory
        self.memory[last_edge_index] = (self.n, self.m, 0,0,0,0,0,0,0,False)

    def numpy_path(self):
        return np.vstack([self.memory["i"], self.memory["j"]]).T

    def get_edge_index(self, i, j):
        """ Suboptimal implementation to get index of an edge """
        graph = self.memory[self.memory['used']]
        a = np.equal(i, graph['i'])
        b = np.equal(j, graph['j'])
        cond = np.logical_and(a, b)
        return np.where(cond)[0][0]

    def edge_exists(self, i, j):
        graph = self.memory[self.memory['used']]
        a = np.equal(i, graph['i'])
        b = np.equal(j, graph['j'])
        cond = np.logical_and(a, b)
        return  np.any(cond)

    def get_edges_of_type(self, edge_type):
        graph = self.memory[self.memory['used']]
        edges = graph[['i', 'j']]
        res = edges[graph['edge_type'] == edge_type]
        return res

    def get_edges(self):
        graph = self.memory[self.memory['used']]
        return graph[['i', 'j']]

