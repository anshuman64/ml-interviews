

def topologicalSort(graph):
    visited = [False] * len(graph)
    degree = [len(graph[v]) for v in graph]
    to_return = []

    def topologicalSortHelper(v):
        visited[v] = True

        for w in graph[v]:
            degree -= 1
            if degree[w] == 0 and not visited[w]:
                topologicalSortHelper(w)

        to_return.append(v)

    for v in graph:
        if degree[v] == 0 and not visited[v]:
            topologicalSortHelper(v)

    return to_return[::-1]
