from scipy.spatial.distance import cdist
import numpy as np

########################
# k-Nearest Neighbors
########################

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

    # Return rounded mean of y neighbors
    return round(y[neighbor_indices].mean())


test_point = [2.7810836, 2.550537003]  # 0
print(knn_classification(x, y, test_point, 3))


########################
# k-Means
########################

def kmeans(x, k, no_of_iterations):
    # Randomly choose centroids
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx]

    # Find distances to centroids
    distances = cdist(x, centroids, 'euclidean')

    # Assign points to centroids
    assignments = np.array(map(np.argmin, distances))

    # Repeat for set # of iterations
    for _ in range(no_of_iterations):
        # Updating centroids as mean of cluster
        centroids = [x[assignments == i].mean(0) for i in range(k)]
        centroids = np.vstack(centroids)

        # Find distances & assign
        distances = cdist(x, centroids, 'euclidean')
        assignments = np.array(map(np.argmin, distances))

    return assignments


########################
# Linear Regression
########################

# Load data
data = np.loadtxt('wine.data', delimiter=',')
np.random.shuffle(data)

# Define X & Y
T = data.copy()[:, 0]
X = data.copy()[:, 1:]

# Normalize data
X = X - X.mean(0)
X = X / X.std(0)

# Add column of ones
X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

# Initialize hyperparameters
n_epoch = 100
l_rate = 1

# Initialize losses & weights
losses = np.zeros(n_epoch)
w = np.random.randn(X.shape[1])


def MSE(T, Y):
    return -np.mean((T-Y)**2)


for m in range(n_epoch):
    # Forward pass
    Y = np.dot(w, X.T)

    # Report metrics
    losses[m] = MSE(T, Y)

    # Backward pass
    delta = Y - T

    # Batch gradient descent
    w = w - l_rate * np.dot(X.T, delta) / len(X)

########################
# Logistic Regression
########################

# Load data
data = np.loadtxt('wine.data', delimiter=',')
np.random.shuffle(data)

# Define X & Y
T = data.copy()[:, 0]
X = data.copy()[:, 1:]

# Normalize data
X = X - X.mean(0)
X = X / X.std(0)

# Add column of ones
X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

# Initialize hyperparameters
n_epoch = 100
l_rate = 1

# Initialize losses & weights
losses = np.zeros(n_epoch)
w = np.random.randn(X.shape[1])


def Sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0/(1.0 + np.exp(-1.0 * z))


def LogisticLoss(T, Y):
    return -np.mean(T*(np.log(Y)) - (1-T)*np.log(1-Y))


for m in range(n_epoch):
    # Forward pass
    Y = Sigmoid(np.dot(w, X.T))
    preds = np.array([1 if Y[i] > 0.5 else 0 for i in range(len(Y))])

    # Report metrics
    losses[m] = LogisticLoss(T, Y)

    # Backward pass
    delta = Y - T

    # SGD
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in indices:
        w = w - l_rate * np.dot(X[i].T, delta[i])


#########################
# Multilayer Perceptron
#########################

def Relu(x):
    return np.maximum(x, 0)


def GradRelu(x):
    return 1. * (x > 0)


def Softmax(x):
    z = x - np.max(x)  # Handle overflow condition
    y = np.exp(z) / np.sum(np.exp(z))
    return y


class MultilayerPerceptron():
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def _initialize_weights(self, X, T):
        _, n_features = X.shape
        _, n_outputs = T.shape

        # Hidden layer
        self.W = np.random.randn(n_features, self.n_hidden)
        # Output layer
        self.V = np.random.randn(self.n_hidden, n_outputs)

    def fit(self, X, T):
        self._initialize_weights(X, T)

        for i in range(self.n_iterations):
            ### Forward Pass ###
            # Hidden Layer
            hidden_input = np.dot(self.W, X)
            Z = Relu(hidden_input)

            # Output Layer
            Y = Softmax(np.dot(self.V, Z))

            ### Backward Pass ###
            # Output Layer
            # normalize delta by 1 / (n_classes * n_examples)
            delta = (T - Y) / (Y.shape[0] * Y.shape[1])
            d_v = np.dot(Z.T, delta)
            d_z = np.dot(self.V.T, delta)

            # Hidden Layer
            delta = GradRelu(hidden_input) * d_z  # activation
            d_w = np.dot(X.T, delta)

            ### Weight Updates ###
            self.V -= self.learning_rate * d_v
            self.W -= self.learning_rate * d_w


########################
# PyTorch Example
########################

# Define AE model class
class LinearAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Single linear encoder layer
        self.encoder = torch.nn.Linear(97, 3)

        # Single linear decoder layer
        self.decoder = torch.nn.Linear(3, 97)

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X

    def encode(self, X):
        return self.encoder(X)


# Define model, optimizer, and criterion
model = LinearAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()

# Train model
epochs = 10000
losses = np.zeros(epochs)

for epoch in range(epochs):
    loss = 0

    # Reset gradients
    optimizer.zero_grad()

    # Run forward pass
    outputs = model(X)

    # Compute & save loss
    train_loss = criterion(outputs, X)
    loss += train_loss.item() / len(X)
    losses[epoch] = loss

    # Backpropagate gradients
    train_loss.backward()

    # Update weights
    optimizer.step()
