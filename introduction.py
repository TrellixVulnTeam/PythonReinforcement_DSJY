import numpy as np

class Network(object):
    def __init__(self, arch=(2, 2, 2, 1)):
        self.arch = arch
        self.weights = self.initializeWeights()
        self.biases = self.initializeBiases()

    def initializeBiases(self):
        biasesSize = [self.arch[i] for i in range(1, len(self.arch))]
        biases = [np.random.uniform(-5, 5, biasesSize[i]).reshape((biasesSize[i], 1)) for i in range(len(biasesSize))]
        return biases

    def initializeWeights(self):
        weightsSize = [(self.arch[i + 1], self.arch[i]) for i in range(len(self.arch) - 1)]
        weights = [np.random.uniform(0, 1, weightsSize[i][0] * weightsSize[i][1]).reshape(weightsSize[i]) for i in range(len(weightsSize))]
        return weights

    def feedForward(self, inputVector):
        activations = [inputVector]
        zs = []
        for i in range(len(self.arch) - 1):
            zVector = self.weights[i] @ activations[i] + self.biases[i]
            zs.append(zVector)
            activationVector = self.activationFunction(zVector)
            activations.append(activationVector)
        return [activations, zs]

    def finalLayerError(self, output, correct, outputZs):
        return ((output - correct) * self.activationDerivative(outputZs))

    def layerErr(self, nextErr, zVector, weights):
        return  (weights.T @ nextErr * self.activationDerivative(zVector))

    def 

    def backpropagation(self):


    def activationDerivative(self, z):
        return self.activationFunction(z) * (1 - self.activationVector(z))

    def activationFunction(self, z):
        return (1 / (np.exp(-z) + 1));

def main():
    student = Network(arch=(2, 2, 2, 1))
    student.feedForward(np.array([[1], [1]]))

if __name__ == "__main__":
    main()
