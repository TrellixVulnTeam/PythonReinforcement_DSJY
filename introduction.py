import numpy as np

class StudentAgent(object):
    def __init__(self):
        # NOTE: 1 extra element in weight and biases matrices. Element at (1, 1)
        self.weights = self.random2x2()
        self.biases = self.random2x2()

    def random2x2(self):
        return np.array([[1.0, 0.0] for _ in range(2)])
        return np.array([np.random.uniform(0, 1, 2) for _ in range(2)])

    def askQuestion(self, x1, x2):
        # Init activations.
        activations = np.zeros((2, 3))
        activations[0][0] = x1
        activations[1][0] = x2

        activations = activations.T

        # Init z values.
        zs = np.zeros((2, 2))
        # Loop through the levels
        for l in range(0, len(self.weights)):
            # Without sigmoid.
            withoutBias = activations[l] @ self.weights
            z = activations[l] @ self.weights + self.biases[l]
            # zs[0][l] = z[0]
            # zs[1][l] = z[1]
            zs[l] = z
            # Sigmoid function.
            a = self.activationFunction(z)
            activations[l + 1] = a
            # activations[0][l + 1] = a[0]
            # activations[1][l + 1] = a[1]

        print(activations)
        # Final result
        result = activations[-1][0]
        return result

    def activationFunction(self, z):
        return (1 / (np.exp(-z) + 1));

def main():
    student = StudentAgent()
    result = student.askQuestion(1.0, 2.0)
    print(result)

if __name__ == "__main__":
    main()
