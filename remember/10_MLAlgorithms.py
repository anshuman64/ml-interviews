########################
## k-Nearest Neighbors
########################
import numpy as np

dataset = np.array([
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]])

x = dataset[:,:2]
y = dataset[:,-1]

def euclidean_dist(x1, x2):
    return np.sqrt( ((x1-x2)**2).sum() )

def get_k_neighbors(x, test_point, k):
    distances = []

    for example in x:
        distances.append(euclidean_dist(example, test_point))

    # Return the indices to k smallest distances
    return np.argsort(distances)[:k]

def knn_classification(x, y, test_point, k):
    neighbor_indices = get_k_neighbors(x, test_point, k)

    return round(y[neighbor_indices].mean())

test_point = [2.7810836,2.550537003] # 0
print(knn_classification(x, y, test_point, 3))


########################
## Logistic Regression
########################

# Load data
data = np.loadtxt('wine.data', delimiter=',')
np.random.shuffle(data)

# Define X & Y
Y = data.copy()[:,0]
X = data.copy()[:,1:]

# Normalize data
X = X - np.average(X, axis=0) 
X = X / np.std(X, axis=0)

# Add column of ones
X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X)) 

# Initialize hyperparameters
n_epoch = 100
l_rate = 1

# Initialize losses & weights
losses = np.zeros(n_epoch)
w = np.random.randn(X.shape[1])

def sigmoid(z):
    z = np.clip(z, -500, 500 )
    return 1.0/(1.0 + np.exp(-1.0 * z))

for m in range(n_epoch):
    # Forward pass
    probas = sigmoid(np.dot(w, X.T))
    preds = np.array([1 if probas[i] > 0.5 else 0 for i in range(len(probas))])

    # Report metrics
    losses[m] = sklearn.metrics.log_loss(Y, probas)

    # Backward pass
    delta = probas - Y

    # SGD
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    for i in indices:
        w = w - l_rate * np.dot(X[i].T, delta[i])


########################
## PyTorch Example
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


#############################
## Classification w/ Generative Model
#############################

K = 8 # number of classes
N = 91 # number of trials per class
D = 97 # number of dimensions

# Calculate MLE statistics
pis = np.full(K, 1/K) # Prior is uniform over all the classes
mus = X_train.mean(axis=1) # Sample mean
covs = np.array([np.dot((X_train[k] - mus[k]).T, X_train[k] - mus[k]) / N for k in range(K)]) # Sample covariance
sigmas = np.full((K, D, D), covs.sum(axis=0) / K) # Weighted average of covariances, same for all classes

# Calculate log likelihood
probas_train = np.array([multivariate_normal.logpdf(X_train_flat, mus[k], sigmas[k]) + np.log(pis[k]) for k in range(K)])

# Calculate predictions
preds_train = np.argmax(probas_train, axis=0)

# Print train accuracy
sklearn.metrics.accuracy_score(preds_train, Y)


########################
## Gaussian Mixture Model - EM Algorithm
########################    
    
# Create GMM class
class GMM:
    """
    Properties:
        mu: means of clusters
        sigma: covariance matrices of clusters
        pi: priors of clusters
        log_likelihoods: log likelihoods per epoch
        gamma: gamma of each example
    """
    def __init__(self, mu, sigma, pi):
        """
        Args:
            mu: initial means of clusters
            sigma: initial covariance matrices of clusters
            pi: initial priors of clusters
        """
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        
        # Infer K (number of clusters) and D (number of dimensions) from mu
        self.K = mu.shape[0]
        self.D = mu.shape[1]
        
    def fit(self, X):
        """
        Args:
            X: data to run the GMM model to
        Returns:
            log_likelihoods: array of log_likelihoods per epoch
        """
        # Save number of examples from X
        self.T = X.shape[0]
        
        # Calculate initial log likelihood
        self.log_likelihoods = np.array(np.sum(np.log(np.sum(np.exp([multivariate_normal.logpdf(X, self.mu[k], self.sigma[k])+np.log(self.pi[k]) for k in range(self.K)]),axis=0)))) / self.T
        
        # Save counter variable
        i = 1
        while(True):
            # E-step - forward pass
            theta_numerator = np.array([multivariate_normal.logpdf(X, self.mu[k], self.sigma[k]) + np.log(self.pi[k]) for k in range(self.K)])
            theta_denominator = np.log(np.sum([np.exp(multivariate_normal.logpdf(X, self.mu[k], self.sigma[k]) + np.log(self.pi[k])) for k in range(self.K)], axis=0))

            self.gamma = np.exp(theta_numerator - theta_denominator).T

            # M-step - backward pass
            N = np.sum(self.gamma, axis=0)

            self.mu = np.dot(self.gamma.T, X) / N[:,np.newaxis]
            self.sigma = np.array([np.sum([self.gamma[t][k] * np.outer((X[t] - self.mu[k]), (X[t] - self.mu[k])) for t in range(self.T)], axis=0)/N[k] for k in range(self.K)]) 
            self.pi = N / self.T

            # Calculate likelihood - loss
            self.log_likelihoods = np.append(self.log_likelihoods, np.sum(np.log(np.sum(np.exp([multivariate_normal.logpdf(X, self.mu[k], self.sigma[k])+np.log(self.pi[k]) for k in range(self.K)]),axis=0))) / self.T) 
            
            # Stopping condition if difference is less than threshold or max iterations reached
            if np.abs(self.log_likelihoods[i] - self.log_likelihoods[i-1]) < 1e-10 or i > 100:
                break
            
            i += 1
        
        return self.log_likelihoods
    
    def predict(self, X):
        """
        Args:
            X: data to fit GMM model to
        Returns:
            predictions: predictions calculated as highest probability cluster
        """
        theta_numerator = np.array([multivariate_normal.logpdf(X, self.mu[k], self.sigma[k]) + np.log(self.pi[k]) for k in range(self.K)])
        theta_denominator = np.log(np.sum([np.exp(multivariate_normal.logpdf(X, self.mu[k], self.sigma[k]) + np.log(self.pi[k])) for k in range(self.K)], axis=0))

        self.gamma = np.exp(theta_numerator - theta_denominator).T
        
        return np.argmax(self.gamma, axis=1)
    
    def calculate_loss(self, X):
        """
        Args:
            X: data to calculate loss for
        Returns:
            log_likelihood: log_likelihood using model
        """
        return np.sum(np.log(np.sum(np.exp([multivariate_normal.logpdf(X, self.mu[k], self.sigma[k])+np.log(self.pi[k]) for k in range(self.K)]),axis=0))) / self.T


####################
## Viterbi Algorithm
#####################

# Load data
pi = pd.read_table('initialStateDistribution.txt', header=None)[0]
A = pd.DataFrame(np.loadtxt('transitionMatrix.txt'))
B = pd.read_table('emissionMatrix.txt', header=None)
O = pd.Series(np.loadtxt('observations.txt'))

# Initialize variables
T = len(O)
n = len(pi)

# Forward pass
L = pd.DataFrame(np.zeros((n, T)))
L[0] = [np.log(pi.iloc[i]) + np.log(B[O[0]].iloc[i]) for i in range(n)]
for t in range(0, T-1):
    L[t+1] = [max(L[t]+np.log(A[j])) + np.log(B[O[t+1]].iloc[j]) for j in range(n)]

# Backward pass
S = pd.Series(np.zeros(T))
S[T-1] = np.argmax(L[T-1])
for t in range(T-2, -1, -1):
    S[t] = np.argmax(L[t]+np.log(A[S[t+1]]))


####################
## Policy Iteration
#####################

# Define gamma
gamma = 0.99

# Load rewards
R = np.loadtxt('rewards.txt')

# Load transition probabilities
PA1 = np.zeros((81,81))
f = open("prob_a1.txt", "r")
for x in f:
    i,j,p = x.split()
    PA1[int(i)-1][int(j)-1] = p

# Concatenate transition probabilities into a single array for easy access
P = [PA1, PA2, PA3, PA4]

# Policy Iteration
# Initialize policy to random values
pi = np.random.randint(0,1,81)

# Calculate P_pi
P_pi = np.zeros((81,81))

for s in range(81):
    for s_prime in range(81):
        P_pi[s][s_prime] = P[pi[s]][s][s_prime]

# Calculate V_pi
V_pi = np.dot(np.linalg.inv(np.identity(81) - gamma * P_pi), R)

# Calculate Q
Q = np.zeros((81,4))
for s in range(81):
    for a in range(4):
        Q[s][a] = sum([P[a][s][s_prime]*V_pi[s_prime] for s_prime in range(81)])

# Calculate pi_prime
pi_prime = np.zeros(81)
for s in range(81):
    pi_prime[s] = np.argmax(Q[s])

# Loop until convergence
while(True):
    pi = pi_prime.astype(int)

    # Calculate P_pi
    for s in range(81):
        for s_prime in range(81):
            P_pi[s][s_prime] = P[pi[s]][s][s_prime]

    # Calculate V_pi
    V_pi_old = V_pi
    V_pi = np.dot(np.linalg.inv(np.identity(81) - gamma * P_pi), R)

    if np.array_equal(V_pi_old,V_pi):
        break

    # Calculate Q
    for s in range(81):
        for a in range(4):
            Q[s][a] = R[s] + gamma * sum([P[a][s][s_prime]*V_pi[s_prime] for s_prime in range(81)])

    # Calculate pi_prime
    for s in range(81):
        pi_prime[s] = np.argmax(Q[s]).astype(int)


####################
## Value Iteration
#####################

# Initialize variables
V = np.ones(81) # will be replaced with zeros
Q = np.zeros((81,4))
V_new = np.zeros(81)

# Loop while the biggest difference between the last iteration and this one is > 0.1
while (max(np.abs(V_new - V)) > 0.000001):
    V = V_new.copy()
    
    for s in range(81):
        for a in range(4):
            # Calculate Q
            Q[s][a] = R[s] + gamma * sum([P[a][s][s_prime]*V[s_prime] for s_prime in range(81)])

        # Calculate V_new
        V_new[s] = max(Q[s])