from dis import dis
import numpy as np
import matplotlib.pyplot as plt
import random

# domain D = [0, 100] x [0, 100]
D = np.zeros((100, 100))
q_init = (50, 50)
delta = 1
K = 0


class Node:
    def __init__(self, pos, children = None):
        if children is None:
            self.children = []
        else:
            self.children = children

        self.pos = pos
    
    def add_child(self, child):
        self.children.append(child)
    
    def remove_child(self, child):
        self.children.remove(child)

    def num_children(self):
        return len(self.children)

    # finds distance between two nodes
    def get_distance(self, other):
        dist = ( (self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2 )**0.5
        return dist


class Graph:
    def __init__(self):
        self._graph = {}
    
    def add_node(self, node):
        self.graph[node] = node.pos
    
    # adds nondirected edge between node1 and node2
    def add_edge(self, node1, node2):
        self._graph[node1].add(node2)
        self._graph[node2].add(node1)

    def remove_node(self, node):
        for n, connections in self._graph.items():
            try:
                connections.remove(node)
            except KeyError:
                pass
        try:
            self._graph.pop(node)
        except KeyError:
            pass

G = Graph()
start_node = Node(q_init)
G.add_node(start_node)

def nearest_vertex(self, q_rand, graph):
    distances = {}
    for node in graph:
        dist = node.get_distance(q_rand)
        distances[node] = dist
    return min(distances, key=distances.get)
    
    
node_vals = []  # contains all positions of nodes - graph G
start_node = Node(q_init)
node_vals.append(start_node.pos)
K = len(start_node.children) + 1
print(K)


q_rand = Node(random.random()*100)
q_near = nearest_vertex(q_rand, node_vals)


xs = [x[0] for x in node_vals]
ys = [x[1] for x in node_vals]
plt.axis([0, 100, 0, 100])
plt.plot(xs, ys, marker='o', markersize=7)
plt.show()