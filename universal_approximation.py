import numpy as np
import matplotlib.pyplot as plt

# 1. Définition de la fonction cible
def target_function(x):
    return -np.arccosh(x)

# 2. Réseau de neurones avec une seule couche cachée et fonction d'activation ReLU
class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_neurons, output_dim):
        # Initialisation aléatoire des poids et biais
        self.W1 = np.random.randn(hidden_neurons, input_dim) * 0.1
        self.b1 = np.random.randn(hidden_neurons, 1) * 0.1
        self.W2 = np.random.randn(output_dim, hidden_neurons) * 0.1
        self.b2 = np.random.randn(output_dim, 1) * 0.1
    
    # On choisit ReLU comme fonction d'activation
    def relu(self, x):
        return np.maximum(0, x)
    
    # Propagation avant pour l'entrainement
    def forward(self, x):
        # Forward propagation
        x = x.reshape(1, -1)
        # Propagation avant
        self.Z1 = self.W1 @ np.log(x) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        return self.Z2.flatten()

    # Backpropagation pour l'entrainement
    def backpropagate(self, X, y, lr=0.01):
        # calcul des gradients et mise à jour des poids
        output = self.forward(X)  # Calcul de la sortie
        loss = (output - y) ** 2  # Erreur quadratique

        # Calcul des gradients
        grad_output = 2 * (output - y)  # Gradient de la sortie
        grad_W2 = grad_output * self.A1.T  # Gradient des poids de la couche de sortie
        grad_b2 = grad_output  # Gradient du biais de la couche de sortie
        grad_A1 = grad_output * self.W2.T  # Propagation du gradient vers la couche cachée
        grad_Z1 = grad_A1 * (self.Z1 > 0)  # ReLU backpropagation
        grad_W1 = grad_Z1 @ X.reshape(1, -1)  # Gradient des poids de la couche cachée
        grad_b1 = grad_Z1  # Gradient du biais de la couche cachée

        # Mise à jour des paramètres
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

        return loss

    # A utiliser quand on veut utiliser le réseau pour prédire un résultat. 
    def predict(self, x):
        return self.forward(x)
        
        

# 3. Entraînement du réseau
def train(network, X, y, epochs=1000, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0
        # Boucle pour chaque exemple de l'ensemble d'entraînement
        for i in range(X.shape[0]):
            # Appel de la fonction de backpropagation pour chaque échantillon
            loss = network.backpropagate(X[i], y[i], lr)
            total_loss += loss
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / X.shape[0]}")


# 4. Visualisation de l'approximation
def plot_approximation(network, X, y):
    y_pred = np.array([network.predict(x) for x in X])
    plt.plot(X, y, label='True Function')
    plt.plot(X, y_pred, label='NN Approximation', linestyle='dashed')
    plt.legend()
    plt.show()

# Exécution de l'illustration

# Générer des données d'entraînement
X = np.linspace(1, 2*np.pi, 100)
Y = target_function(X)

# Créer le réseau de neurones et l'entraîner
hidden_neurons = 10 # Essayez d'augmenter ce nombre pour voir l'approximation s'améliorer
network = SimpleNeuralNetwork(input_dim=1, hidden_neurons=hidden_neurons, output_dim=1)
train(network, X, Y, epochs=2000, lr=0.01)


# Visualisation l'approximation
# Affiche la courbe de la fonction réelle, et la courbe "prédite" par le réseau de neurone. 
plot_approximation(network, X, Y)
