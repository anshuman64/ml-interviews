from collections import deque
import heapq
from queue import PriorityQueue
import numpy as np


stack = __import__('4_Stack')
queue = __import__('5_Queue')


###############
# DFS & BFS
###############

class Graph:
    def __init__(self):
        self.children = {}

    def add_vertex(self, name):
        self.children[name] = []

    def add_edge(self, start_vertex, end_vertex):
        self.children[start_vertex].append(end_vertex)

    def dfsRecursive(self, root_node):
        visited = set()

        def dfsHelper(node):
            # Visit node
            print(node)
            visited.add(node)

            # Add univisited children
            for child_node in node.children:
                if child_node not in visited:
                    dfsHelper(child_node)

        dfsHelper(root_node)

    def depthFirstTraversal(self, root_node):
        node_stack = []
        visited = set()

        node_stack.append(root_node)
        visited.add(root_node)

        while node_stack:
            current_node = node_stack.pop()
            print(current_node)

            # Add unvisited children
            for child_node in current_node.children:
                if child_node not in visited:
                    node_stack.push(child_node)
                    visited.add(child_node)

    def breadthFirstTraversal(self, root_node):
        node_queue = deque()
        visited = set()

        node_queue.append(root_node)
        visited.add(root_node)

        while node_queue:
            current_node = node_queue.popleft()
            print(current_node)

            # Add unvisited children
            for child_node in current_node.children:
                if child_node not in visited:
                    node_queue.append(child_node)
                    visited.add(child_node)


###############
# Representations
###############

class GraphAdjacencyList:
    def __init__(self):
        self.adjacencies = {}

    def add_vertex(self, name):
        self.adjacencies[name] = list()

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex].append(end_vertex)


class GraphAdjacencyMatrix:
    def __init__(self, num_vertices):
        self.adjacencies = [
            [0 for _ in range(num_vertices)] for _ in range(num_vertices)]

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex][end_vertex] = 1


###############
# Dijsktras
###############

def dijskstras(graph, start_vertex):
    # O(V+ElogE)
    # Source: https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/
    # Source: https://stackoverflow.com/questions/9255620/why-does-dijkstras-algorithm-use-decrease-key

    N = len(graph)
    # Initialize best distances to each vertex as inf
    distances = {vertex: np.Inf for vertex in graph}
    # Initialize best path previous vertex to each vertex as None
    previous = {vertex: None for vertex in graph}

    # Initialize start vertex best distance as 0
    distances[start_vertex] = 0
    # Initialize priority queue with start_vertex
    priority_queue = [(0, start_vertex)]

    while len(priority_queue) > 0:
        # Pop lowest distance vertex in priority queue
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Only consider vertex if it hasn't been visited before
        if current_distance > distances[current_vertex]:
            continue

        # For each edge from vertex
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            # If distance is less than saved best distance
            if distance < distances[neighbor]:
                # Update best distance
                distances[neighbor] = distance

                # Update previous vertex
                previous[neighbor] = current_vertex

                # Add to priority queue
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous


####################
# Topological Sort
####################

def topologicalSort(graph):
    # Initialize starting degrees
    degree = [len(graph[v].children) for v in graph]
    # Initialize to_return
    to_return = []

    q = deque([v for v in degree if degree[v] == 0])
    while q:
        v = q.popleft()
        to_return.append(v)

        for neighbor in graph[v]:
            degree[neighbor] -= 1

            if degree[neighbor] == 0:
                q.append(neighbor)

    return to_return


###############
# Main
###############

def test_graph():
    g = Graph()
    g.add_vertex(0)
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    g.add_vertex(4)
    g.add_vertex(5)

    g.add_edge(0, 1)
    g.add_edge(0, 4)
    g.add_edge(0, 5)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(2, 1)
    g.add_edge(3, 2)
    g.add_edge(3, 4)

    g.dfsRecursive(g.nodes[0])


if __name__ == '__main__':
    test_graph()
