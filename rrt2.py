from dis import dis
from hashlib import new
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import random

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

    # def num_children(self):
    #     return len(self.children)

    # finds distance between two nodes
    def get_distance(self, other):
        dist = ( (self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2 )**0.5
        return dist


class Graph:
    def __init__(self):
        self.graph = {}
        self.connections = []
        self.cxn_pos = []
        self._length = 0
    
    def add_node(self, node):
        self.graph[node] = node.pos
        self._length += 1
    
    # adds nondirected edge between node1 and node2
    def add_edge(self, node1, node2):
        node1.children.append(node2)
        node2.children.append(node1)
        self.connections.append((node1, node2))
        self.cxn_pos.append((node1.pos, node2.pos))

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


class Obstacle:
    def __init__(self, radius, center_pos):
        self.radius = radius
        self.center = center_pos    # center_pos is an (x,y) tuple

    # checks if there is a collision between the obstacle and the path between the two nodes
    def collision(self, node1, node2):
        # slope of line btw nodes
        m = (node2.pos[1] - node1.pos[1])/(node2.pos[0] - node1.pos[0])

        # find perpendicular point
        x = (self.center[0]/m + self.center[1] + m*node1.pos[0] - node1.pos[1]) / (m + 1/m)
        y = m*(x-node1.pos[0]) + node1.pos[1]


        # check if either node is inside the obstacle
        nodes = [node1, node2]
        for node in nodes:
            if (node.pos[0] - self.center[0])**2 + (node.pos[1] - self.center[1])**2 <= self.radius**2:
                return True

        # check if perpendicular point is between the two nodes
        # if yes, check if collision
        if ( 
            (min(node1.pos[0], node2.pos[0]) < x) and 
            (x < max(node1.pos[0], node2.pos[0]) ) and 
            (min(node1.pos[1], node2.pos[1]) < y) and
            (y < max(node1.pos[1], node2.pos[1])) 
            ):
            # print('between nodes')
            dist = ( (x - self.center[0])**2 + (y - self.center[1])**2 )**0.5
            if self.radius >= dist:
                # print('collision')
                return True

        return False


# returns q_rand = a randomly generated position in the domain (format [xmin, xmax, ymin, ymax])
def random_configuration(domain):
    q_rand = Node((random.uniform(domain[0], domain[1]), random.uniform(domain[2], domain[3])))
    return q_rand

# returns q_near = the node that is closeset to q_rand
def nearest_vertex(q_rand, graph):
    distances = {}
    for node in graph.graph:
        dist = node.get_distance(q_rand)
        distances[node] = dist
        # print(node, distances[node])

    # q_near = min(distances, key=distances.get)
    min_dist = min(distances.values())
    # print(min_dist)
    for key, value in distances.items():
        if value == min_dist:
            q_near = key

    # print(q_near)
    return q_near

# returns q_new = the new node created by moving distance delta from q_near in the direction of q_rand
def new_configuration(q_near, q_rand, delta):
    vector = [q_rand.pos[0] - q_near.pos[0], q_rand.pos[1] - q_near.pos[1]]
    magnitude = ( vector[0]**2 + vector[1]**2 )**0.5
    delta_unit_vector = [delta * x / magnitude for x in vector]     # delta * unit vector
    new_pos = (q_near.pos[0] + delta_unit_vector[0], q_near.pos[1] + delta_unit_vector[1])
    q_new = Node(new_pos)
    return q_new
    


# domain D = [0, 100] x [0, 100]
# [xmin, xmax, ymin, ymax]
D = [0, 100, 0, 100]
q_init = (50, 50)
delta = 5
K = 500

G = Graph()
start_node = Node(q_init)
G.add_node(start_node)

obs1 = Obstacle(5, (25,25))
obs2 = Obstacle(5, (75,75))
obs_list = []
obs_list.append(obs1)
obs_list.append(obs2)

obs_x = [obs.center[0] for obs in obs_list]
obs_y = [obs.center[1] for obs in obs_list]
obs_r = [obs.radius for obs in obs_list]


# build tree into graph given K, D, G, and delta
def build_tree(num_vertices, domain, graph, delta):
    K = num_vertices
    D = domain
    G = graph

    # *** fix issue where if skipping a node, don't end up with less nodes (maybe do K+=1)
    # for i in range(K):
    i = 0
    while i <= K:
        q_rand = random_configuration(D)
        q_near = nearest_vertex(q_rand, G)
        q_new = new_configuration(q_near, q_rand, delta)

        # check for collisions
        for obs in obs_list:
            if (obs.collision(q_near, q_new) ==  True):
                break   # if there is any collision, we want to skip this node
        else:   # executes if for loop did not break
            G.add_node(q_new)
            G.add_edge(q_near, q_new)
            i += 1
            continue

    return G

G = build_tree(K, D, G, delta)


xs = [node.pos[0] for node in G.graph]
ys = [node.pos[1] for node in G.graph]

# creating figure
fig, ax = plt.subplots()
plt.axis(D)
# plotting tree
plt.plot(xs, ys, marker='o', markersize=3, linestyle = 'None')

# plotting obstacles
for obs in obs_list:
    circle = Circle(obs.center, obs.radius, color='green')
    ax.add_patch(circle)

# plotting edges
segments = [cxn for cxn in G.cxn_pos]
line_segments = LineCollection(segments)
ax.add_collection(line_segments)

# making figure square and displaying
plt.gca().set_aspect('equal', adjustable='box')
plt.show()