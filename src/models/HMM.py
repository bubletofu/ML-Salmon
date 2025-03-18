import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score

class HMM:
    def __init__(self, n_components, n_features, random_state=None):
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)

        # Initialize model parameters
        self.start_prob = np.ones(self.n_components) / self.n_components
        self.trans_prob = np.ones((self.n_components, self.n_components)) / self.n_components
        self.emission_prob = np.ones((self.n_components, self.n_features)) / self.n_features

    def fit(self, X, max_iter=100):
        # Implementing the Baum-Welch algorithm (EM)
        for _ in range(max_iter):
            # E-step: Compute the forward and backward probabilities
            alpha, scale_factors = self._forward(X)
            beta = self._backward(X, scale_factors)
            
            # M-step: Update model parameters using the computed probabilities

            self._update_params(X, alpha, beta, scale_factors)

    def _forward(self, X):
        N = self.n_components
        T = X.shape[0]
        
        # Initialize alpha
        alpha = np.zeros((T, N))
        scale_factors = np.zeros(T)
        
        # Initial alpha (probability of first observation given the initial state)
        alpha[0, :] = self.start_prob * self._emission_prob(X[0])
        scale_factors[0] = np.sum(alpha[0, :])
        alpha[0, :] /= scale_factors[0]
        
        # Forward pass
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.trans_prob[:, j]) * self._emission_prob(X[t])[j]
            scale_factors[t] = np.sum(alpha[t, :])
            alpha[t, :] /= scale_factors[t]
        
        return alpha, scale_factors

    def _backward(self, X, scale_factors):
        N = self.n_components
        T = X.shape[0]
        
        # Initialize beta
        beta = np.zeros((T, N))
        
        # Initial beta (backward probability)
        beta[T-1, :] = 1 / scale_factors[T-1]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(self.trans_prob[i, :] * self._emission_prob(X[t+1]) * beta[t+1, :])
            beta[t, :] /= scale_factors[t]
        
        return beta

    def _update_params(self, X, alpha, beta, scale_factors):
        N = self.n_components
        T = X.shape[0]
        
        # Update start probabilities
        self.start_prob = alpha[0, :] * beta[0, :] / np.sum(alpha[0, :] * beta[0, :])
        
        # Update transition probabilities
        trans_prob_numer = np.zeros((N, N))
        trans_prob_denom = np.zeros(N)
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    trans_prob_numer[i, j] += alpha[t, i] * self.trans_prob[i, j] * self._emission_prob(X[t+1])[j] * beta[t+1, j]
            for i in range(N):
                trans_prob_denom[i] += np.sum(alpha[t, i] * beta[t, i])
        
        self.trans_prob = trans_prob_numer / trans_prob_denom[:, None]

        # Update emission probabilities
        emission_prob_numer = np.zeros((N, self.n_features))
        emission_prob_denom = np.zeros(N)
        for t in range(T):
            for i in range(N):
                emission_prob_numer[i, :] += alpha[t, i] * beta[t, i] * X[t]
            for i in range(N):
                emission_prob_denom[i] += np.sum(alpha[t, i] * beta[t, i])
        
        self.emission_prob = emission_prob_numer / emission_prob_denom[:, None]

    def _emission_prob(self, X_t):
        # Use the emission probability to calculate the probability of the observed feature given the state
        return np.exp(-0.5 * np.sum((X_t - self.emission_prob) ** 2, axis=1))

    def predict(self, X):
        # Use Viterbi algorithm to predict the most likely sequence of hidden states
        N = self.n_components
        T = X.shape[0]
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        # Initial delta
        delta[0, :] = self.start_prob * self._emission_prob(X[0])
        
        # Viterbi pass
        for t in range(1, T):
            for j in range(N):
                delta[t, j] = np.max(delta[t-1, :] * self.trans_prob[:, j]) * self._emission_prob(X[t])[j]
                psi[t, j] = np.argmax(delta[t-1, :] * self.trans_prob[:, j])
        
        # Backtrack the best path
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path

    def score(self, X, y):
        # Calculate the log-likelihood score for the model
        alpha, _ = self._forward(X)
        return np.sum(np.log(alpha[:, 0]))  # Log-likelihood of the observation sequence

