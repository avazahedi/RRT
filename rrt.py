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
        self.graph = {}
        self.connections = []
        self._length = 0
    
    def add_node(self, node):
        self.graph[node] = node.pos
        self._length += 1
    
    # adds nondirected edge between node1 and node2
    def add_edge(self, node1, node2):
        self.graph[node1].add(node2)
        self.graph[node2].add(node1)
        self.connections.append((node1, node2))

    def remove_node(self, node):
        for n, connections in self.graph.items():
            try:
                connections.remove(node)
            except KeyError:
                pass
        try:
            self.graph.pop(node)
            self._length -= 1
        except KeyError:
            pass
    
    def num_vertices(self):
        return self._length


# returns q_near = the node that is closeset to q_rand
def nearest_vertex(q_rand, graph):
    distances = {}
    for node in graph.graph:
        dist = node.get_distance(q_rand)
        distances[node] = dist
        q_near = min(distances, key=distances.get)
        return q_near

# returns q_new = the new node created by moving distance delta from q_near in the direction of q_rand
def new_configuration(q_near, q_rand, delta):
    vector = [q_rand[0] - q_near[0], q_rand[1] - q_near[1]]
    magnitude = ( vector[0]**2 + vector[1]**2 )**0.5
    unit_vector = [x / magnitude for x in vector]
    new_pos = (q_near[0] + unit_vector[0], q_near[1] + unit_vector[1])
    q_new = Node(new_pos)
    return q_new
    

G = Graph()
start_node = Node(q_init)
G.add_node(start_node)

q_rand = Node((random.random()*100, random.random()*100))
q_near = nearest_vertex(q_rand, G)
print(q_rand.pos, q_near.pos)


xs = [node.pos[0] for node in G.graph]
ys = [node.pos[1] for node in G.graph]
plt.axis([0, 100, 0, 100])
plt.plot(xs, ys, marker='o', markersize=7)
plt.show()