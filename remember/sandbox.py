import numpy as np

dataset = np.array([
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1]])

x = dataset[:, :2]
y = dataset[:, -1]


def euclidean_dist(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())


def get_k_neighbors(x, test_point, k):
    # Calculate euclidean distance for each point
    distances = map(lambda x: euclidean_dist(x, test_point), x)

    # Return the indices to k smallest distances
    return np.argsort(distances)[:k]


def knn_classification(x, y, test_point, k):
    neighbor_indices = get_k_neighbors(x, test_point, k)

    return round(y[neighbor_indices].mean())


test_point = [2.7810836, 2.550537003]  # 0
print(knn_classification(x, y, test_point, 3))
