import numpy as np
import math

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
        difference = output - correct
        return (difference * self.activationDerivative(outputZs))

    def layerErr(self, nextErr, zVector, weights):
        a = np.dot(weights.T, nextErr)
        return  (a * self.activationDerivative(zVector))

    def weightCost(self, previousActivation, currErr):
        return np.dot(previousActivation.T, currErr)

    def backpropagation(self, inputVector, correctOutput):
        activations, zs = self.feedForward(inputVector)
        output = np.array(activations).T[-1]
        lastZs = np.array(zs).T[-1]
        lastError = self.finalLayerError(output, correctOutput, lastZs)
        deltas = [0 for i in range(len(self.arch) - 1)]
        deltas[-1] = lastError
        for L in range(len(deltas) - 2, -1, -1):
            deltas[L] = self.layerErr(deltas[L + 1], zs[L], self.weights[L + 1])
        weightGradient = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        for Y in range(len(self.weights)):
            for j in range(len(self.weights[Y])):
                for k in range(len(self.weights[Y][j])):
                    weightGradient[Y][j][k] = activations[Y][k] * deltas[Y][j]

        biasGradient = [np.zeros(self.biases[i].shape) for i in range(len(self.biases))]
        for U in range(len(deltas)):
            biasGradient[U] = deltas[U]
        return [weightGradient, biasGradient]

    def gradientDescent(self, trainingData, tr):
        weightNabla = [np.zeros(i.shape) for i in self.weights]
        biasNabla = [np.zeros(i.shape) for i in self.biases]
        # Loop trough all samples.
        for sample in trainingData:
            inputVector, correct = sample
            weightGradient, biasGradient = self.backpropagation(inputVector, correct)
            # Update nablas.
            for i in range(len(weightNabla)):
                weightNabla[i] = weightNabla[i] + weightGradient[i]
            for j in range(len(biasNabla)):
                biasNabla[j] = biasNabla[j] + biasGradient[j]

        # Finally update the current weights and biases.
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (tr/len(trainingData) * weightNabla[i])
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] - (tr/len(trainingData) * biasNabla[i])

    def activationDerivative(self, z):
        return self.activationFunction(z) * (1 - self.activationFunction(z))

    def activationFunction(self, z):
        return (1 / (np.exp(-z) + 1));

def simpleFunctionData(size):
    coordinates = [np.random.uniform(0, 10, 2).reshape((2, 1)) for _ in range(size)]
    data = []
    for i in range(size):
        x = coordinates[i][0][0]
        y = coordinates[i][1][0]

        # Function 2*x + 1
        label = 1
        if y < 2 * x + 1:
            label = 0
        data.append([coordinates[i], label])
    return data

def main():
    data = simpleFunctionData(10)
    student = Network(arch=(2, 3, 1))
    for i in range(10000):
        student.gradientDescent(data, 0.2)
    errSum = 0
    testData = simpleFunctionData(10)
    for sample in testData:
        guess = student.feedForward(sample[0])[0][-1]
        correct = sample[1]
        err = (correct - guess)
        errSum = errSum + abs(err[0][0])
    print(errSum / len(testData))
    result = student.feedForward(np.array([[2], [1]]))
    print(result[0][-1][0][0])

if __name__ == "__main__":
    main()
