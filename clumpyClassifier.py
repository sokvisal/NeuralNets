import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient

# This is a binary classification deep neural network models. Designed to classify galaxies as being in clumpy in morphologies or not

class BinaryModel:
    def __init__(self, trainingData, trainingClass, layer_dims = [10,1], keep_prob=1, alpha=0.005, iter_=1000):
        '''
        Arguments:
        trainingData -- input data for training, with shape (input size, number of examples) 
        trainingClass -- input class for training data, with shape (1, number of examples) 
        layer_dims -- list, define number of hidden layers and nodes [1st hidden layer with # of nodes, 2nd hidden layer with # of nodes]
        keep_prob -- float (between 0-1), probability a node is dropped during the training process
        alpha -- float, learning rate
        iter_ -- int, number of iterations
        '''

        input_dim = trainingData.shape[0] 
        self.training_size = trainingData.shape[1] 

        self.layer_dims = [input_dim]+layer_dims 
        self.trainingData = trainingData
        self.trainingClass = trainingClass

        self.keep_prob = keep_prob
        self.alpha = alpha
        self.iter = iter_

        print ('learning rate = {}'.format(self.alpha))
        print ('number of iterations = {}'.format(self.iter))

        self._initialize_parameters(self.layer_dims)

        
    def train(self):

        for i in tqdm(range(self.iter)):
            self._forward_pass(self.layer_dims)
            self._backward_pass(self.layer_dims)
            self._update_parameters()

        print (self.caches['A3'])
        # return self.weights, self.biases

    def get_prediction(self):
        return self.caches['A3']

    def _softmax(self, z, deri=False):
        z_exp = np.exp(z - z.max())
        if deri:
            return z_exp / np.sum(z_exp) * (1 - z_exp / np.sum(z_exp))
        return  z_exp / np.sum(z_exp)

    def _d_softmax(self, z):
        soft_z = self._softmax(z) 
        tmp = - soft_z * soft_z.reshape(self.layer_dims[-1], 1)
        z_deri = np.diag(soft_z) + tmp

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z) )

    def _relu(self, z):
        return np.maximum( 0, z )

    def _loss_func(self, predictions):
        # print (self.trainingClass.shape, predictions.shape)

        loss = -self.trainingClass*np.log(predictions) - (1-self.trainingClass)*np.log(1-predictions)
        loss /= np.nansum(loss)

        self.loss += [ loss ]

    def _initialize_parameters(self, layer_dims):
        '''
        Initialize the parameters for the weights and biases based on the given number of layer dimensions
        '''
        
        self.parameters = {}
        self.loss = []

        # initialize arrays for weights and biases 
        # 0th index represent the input size while 1st index is the first hidden layer
        for i in range(1, len(layer_dims)):
            self.parameters['W{}'.format(i)] = np.random.randn( layer_dims[i] , layer_dims[i-1] ) / np.sqrt(layer_dims[i-1])
            self.parameters['B{}'.format(i)] = np.zeros((layer_dims[i], 1))
            # self.parameters['D{}'.format(i)] = np.random.randn( layer_dims[i] , layer_dims[i-1] ) / np.sqrt(layer_dims[i-1])

        # return self.parameters

    def _drop_out(self, activation_matrix ):
        '''
        Deactivate some nodes in the given activation matrix, in order to achieve higher prediction accuracy
        '''

        D = np.random.rand( activation_matrix.shape[0], activation_matrix.shape[1] )
        D = D<self.keep_prob
        activation_matrix *= D
        # scale value of nodes to account for the fact that some nodes are shut down
        activation_matrix /= self.keep_prob

        return activation_matrix

    def _forward_pass(self, layer_dims):
        '''
        Since the NN is generalized, we need to set up the unique conditions for the first hidden layer and output layer. 
        In the first hidden layer, the weight matrix is dotted by the input data. In the output layer, the activation function is taken to be the sigmoid function.
        The other layers takes the relu function to be the activation function.
        '''
        # define the caches for later use
        self.caches = {}

        for i in range(1, len(layer_dims)):
            
            w = self.parameters['W{}'.format(i)]
            b = self.parameters['B{}'.format(i)]
            
            # use relu as an activation function if it's not the output layer, 
            # otherwise use the sigmoid function
            # note that if keep_prob = 1 then there is no regularization as no nodes are dropped out
            if i == 1:
                self.caches['Z{}'.format(i)] = np.dot(w, self.trainingData) + b
                self.caches['A{}'.format(i)] = self._relu( self.caches['Z{}'.format(i)] )

                self.caches['A{}'.format(i)] = self._drop_out(self.caches['A{}'.format(i)] )

            elif i == len(layer_dims)-1:
                self.caches['Z{}'.format(i)] = np.dot(w, self.caches['A{}'.format(i-1)]) + b
                self.caches['A{}'.format(i)] = self._sigmoid( self.caches['Z{}'.format(i)] ) # _sigmoid
            else:
                self.caches['Z{}'.format(i)] = np.dot(w, self.caches['A{}'.format(i-1)]) + b
                self.caches['A{}'.format(i)] = self._relu( self.caches['Z{}'.format(i)] )

                self.caches['A{}'.format(i)] = self._drop_out(self.caches['A{}'.format(i)]  )

            self.caches['W{}'.format(i)] = self.parameters['W{}'.format(i)]
            self.caches['B{}'.format(i)] = self.parameters['B{}'.format(i)]

        # # calculate and update loss function
        # self._loss_func(self.caches['A{}'.format(i)])

    def _backward_pass(self, layer_dims):
        '''
        Calulate the gradient for all the weights and biases by taking the partial derivative with respect to the loss function.
        Again since a generalized NN is used, unique conditions are set up for the first and output hidden layers.
        '''
        caches = self.caches 
        
        self.gradient = {}

        for i in range(1, len(layer_dims) )[::-1]:

            a_n = self.caches['A{}'.format(i)]
            w_n = self.caches['W{}'.format(i)]

            if i == len(layer_dims)-1:
                # dz_n = a_n - y
                self.gradient['dZ{}'.format(i)] =  a_n - self.trainingClass #shape of (1,training_size)
                # self.gradient['dZ{}'.format(i)] =  2*(a_n - self.trainingClass) /  self.trainingClass.shape[0] * self._softmax(self.caches['Z{}'.format(i)], deri=True)
                # dw_n = dz_n * a_(n-1).T
                self.gradient['dW{}'.format(i)] = np.dot( self.gradient['dZ{}'.format(i)], self.caches['A{}'.format(i-1)].T) / self.training_size
            elif i == 1:
                self.gradient['dZ{}'.format(i)] = np.multiply( self.gradient['dA{}'.format(i)], np.int64(a_n>0) )
                self.gradient['dW{}'.format(i)] = np.dot( self.gradient['dZ{}'.format(i)], self.trainingData.T) / self.training_size
            else:
                self.gradient['dZ{}'.format(i)] = np.multiply( self.gradient['dA{}'.format(i)], np.int64(a_n>0) )
                self.gradient['dW{}'.format(i)] = np.dot( self.gradient['dZ{}'.format(i)], self.caches['A{}'.format(i-1)].T) / self.training_size

            self.gradient['dB{}'.format(i)] = np.sum( self.gradient['dZ{}'.format(i)], axis=1, keepdims=True) / self.training_size
            if i > 1:
                self.gradient['dA{}'.format(i-1)] = np.dot( w_n.T, self.gradient['dZ{}'.format(i)] )
                # deactivate nodes that were take off if any
                self.gradient['dA{}'.format(i-1)] = self._drop_out(self.gradient['dA{}'.format(i-1)]  )

        # print(self.gradient.keys())

    def _update_parameters(self):

        for i in range( 1, len(self.layer_dims) ):
            self.parameters['W{}'.format(i)] -= self.alpha*self.gradient['dW{}'.format(i)]
            self.parameters['B{}'.format(i)] -= self.alpha*self.gradient['dB{}'.format(i)]