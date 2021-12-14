stack = __import__('4_Stack')
queue = __import__('5_Queue')

class GraphNode:
    def __init__(self, name):
        self.name = name
        self.children = []

    def __repr__(self):
        return str(self.name)

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_vertex(self, name):
        self.nodes[name] = GraphNode(name)

    def add_edge(self, start_vertex, end_vertex):
        self.nodes[start_vertex].children.append(self.nodes[end_vertex])

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
        node_stack = stack.Stack()
        visited = set()
        node_stack.push(root_node)

        while (not node_stack.isEmpty()):
            # Pop node and check if visited
            current_node = node_stack.pop()
            if current_node not in visited:
                
                # Visit node
                print(current_node)
                visited.add(current_node)

                # Add unvisited children
                for child_node in current_node.children:
                    if child_node not in visited:
                        node_stack.push(child_node)

    def breadthFirstTraversal(self, root_node):
        node_queue = queue.Queue()
        visited = set()
        node_queue.push(root_node)

        while (not node_queue.isEmpty()):
            # Pop node and check if visited
            current_node = node_queue.pop()
            if current_node not in visited:
                
                # Visit node
                print(current_node)
                visited.add(current_node)

                # Add unvisited children
                for child_node in current_node.children:
                    if child_node not in visited:
                        node_queue.push(child_node)

class GraphAdjacencyList:
    def __init__(self):
        self.adjacencies = {}

    def add_vertex(self, name):
        self.adjacencies[name] = list()

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex].append(end_vertex)

class GraphAdjacencyMatrix:
    def __init__(self, num_vertices):
        self.adjacencies = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]

    def add_edge(self, start_vertex, end_vertex):
        self.adjacencies[start_vertex][end_vertex] = 1


def test_graph():
    g = Graph()
    g.add_vertex(0)
    g.add_vertex(1)
    g.add_vertex(2)
    g.add_vertex(3)
    g.add_vertex(4)
    g.add_vertex(5)

    g.add_edge(0,1)
    g.add_edge(0,4)
    g.add_edge(0,5)
    g.add_edge(1,3)
    g.add_edge(1,4)
    g.add_edge(2,1)
    g.add_edge(3,2)
    g.add_edge(3,4)

    g.dfsRecursive(g.nodes[0])

if __name__ == '__main__':
    test_graph()