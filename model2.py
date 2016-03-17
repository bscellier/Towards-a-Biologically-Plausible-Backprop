import cPickle
from external_world import External_World
import numpy as np
import os
import theano
import theano.tensor as T

def pi(s):
    return T.clip(s, 0., 1.)

class Network(object):

    def __init__(self, name, hyperparameters=dict()):

        self.path = name+".save"

        # LOAD/INITIALIZE PARAMETERS
        self.biases, self.weights, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)

        # LOAD EXTERNAL WORLD (=DATA)
        self.external_world = External_World()
        
        # INITIALIZE PERSISTENT PARTICLES
        dataset_size = self.external_world.size_dataset
        layer_sizes = [28*28] + self.hyperparameters["hidden_sizes"] + [10]
        values = [np.zeros((dataset_size, layer_size), dtype=theano.config.floatX) for layer_size in layer_sizes[1:]]
        self.persistent_particles  = [theano.shared(value, borrow=True) for value in values]

        # LAYERS = MINI-BACTHES OF DATA + MINI-BACTHES OF PERSISTENT PARTICLES
        batch_size = self.hyperparameters["batch_size"]
        self.index = theano.shared(np.int32(0), name='index') # index of a mini-batch

        self.x_data = self.external_world.x[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data = self.external_world.y[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        self.layers = [self.x_data]+[particle[self.index * batch_size: (self.index + 1) * batch_size] for particle in self.persistent_particles]

        # BUILD THEANO FUNCTIONS
        self.change_mini_batch_index = self.__build_change_mini_batch_index()
        self.measure                 = self.__build_measure()
        self.negative_phase          = self.__build_negative_phase()
        self.positive_phase          = self.__build_positive_phase()

    def save_params(self):
        f = file(self.path, 'wb')
        biases_values  = [b.get_value() for b in self.biases]
        weights_values = [W.get_value() for W in self.weights]
        to_dump        = biases_values, weights_values, self.hyperparameters, self.training_curves
        cPickle.dump(to_dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def __load_params(self, hyperparameters):

        hyper = hyperparameters

        # Glorot/Bengio weight initialization
        def initialize_layer(n_in, n_out):
            rng = np.random.RandomState()
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            return W_values

        if os.path.isfile(self.path):
            f = file(self.path, 'rb')
            biases_values, weights_values, hyperparameters, training_curves = cPickle.load(f)
            f.close()
            for k,v in hyper.iteritems():
                hyperparameters[k]=v
        else:
            layer_sizes = [28*28] + hyperparameters["hidden_sizes"] + [10]
            biases_values  = [np.zeros((size,), dtype=theano.config.floatX) for size in layer_sizes]
            weights_values = [initialize_layer(size_pre,size_post) for size_pre,size_post in zip(layer_sizes[:-1],layer_sizes[1:])]
            training_curves = dict()
            training_curves["training error"]   = list()
            training_curves["validation error"] = list()

        biases  = [theano.shared(value=value, borrow=True) for value in biases_values]
        weights = [theano.shared(value=value, borrow=True) for value in weights_values]

        return biases, weights, hyperparameters, training_curves

    # SET INDEX OF THE MINI BATCH
    def __build_change_mini_batch_index(self):

        index_new = T.iscalar("index_new")

        change_mini_batch_index = theano.function(
            inputs=[index_new],
            outputs=[],
            updates=[(self.index,index_new)]
        )

        return change_mini_batch_index

    # ENERGY FUNCTION, DENOTED BY E
    def __energy(self, layers):
        squared_norm    =   sum( [T.batched_dot(layer,layer)       for layer      in layers] ) / 2.
        linear_terms    = - sum( [T.dot(layer,b)                   for layer,b    in zip(layers,self.biases)] )
        quadratic_terms = - sum( [T.batched_dot(T.dot(pre,W),post) for pre,W,post in zip(layers[:-1],self.weights,layers[1:])] )
        return squared_norm + linear_terms + quadratic_terms

    # COST FUNCTION, DENOTED BY C
    def __cost(self, layers):
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(axis=1)

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __build_measure(self):

        E = T.mean(self.__energy(self.layers))
        C = T.mean(self.__cost(self.layers))
        y_prediction = T.argmax(self.layers[-1], axis=1)
        error        = T.mean(T.neq(y_prediction, self.y_data))

        measure = theano.function(
            inputs=[],
            outputs=[E, C, error]
        )

        return measure

    def __build_negative_phase(self):

        n_iterations = T.iscalar('n_iterations')

        def forwardprop(*layers):
            layers_new = [layers[0]]
            for k in range(1,len(self.layers)-1):
                layers_new += [pi(T.dot(layers_new[-1], self.weights[k-1]) + T.dot(layers[k+1], self.weights[k].T) + self.biases[k])]
            layers_new += [pi(T.dot(layers_new[-1], self.weights[-1]) + self.biases[-1])]
            return layers_new
        
        ( layers, updates ) = theano.scan(
            forwardprop,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_end = [layer[-1] for layer in layers]
        
        for particles,layer,layer_end in zip(self.persistent_particles,self.layers[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)

        negative_phase = theano.function(
            inputs=[n_iterations],
            outputs=[],
            updates=updates
        )

        return negative_phase

    def __build_positive_phase(self):

        n_iterations = T.iscalar('n_iterations')
        alphas = [T.fscalar("alpha_W"+str(r+1)) for r in range(len(self.weights))]

        def backprop(*layers):
            layers_new = [layers[-1]]
            for k in range(len(self.layers)-2,0,-1):
                layers_new += [pi(T.dot(layers[k-1], self.weights[k-1]) + T.dot(layers_new[-1], self.weights[k].T) + self.biases[k])]
            layers_new += [layers[0]]
            layers_new.reverse()
            return layers_new

        ( layers, updates ) = theano.scan(
            backprop,
            outputs_info=self.layers[:-1]+[self.y_data_one_hot],
            n_steps=n_iterations
        )
        layers_new = [layer[-1] for layer in layers]

        Delta_layers = [(layer_new-layer) for layer_new,layer in zip(layers_new[1:],self.layers[1:])]
        biases_new  = [b + alpha * T.mean(Delta, axis=0) for b,alpha,Delta in zip(self.biases[1:],alphas,Delta_layers)]
        weights_new = [W + alpha * T.dot(layer.T, Delta) / T.cast(self.layers[0].shape[0], dtype=theano.config.floatX) for W, alpha, layer, Delta in zip(self.weights, alphas, self.layers[:-1], Delta_layers)]
        
        for bias, bias_new in zip(self.biases[1:],biases_new):
            updates[bias]=bias_new
        for weight, weight_new in zip(self.weights,weights_new):
            updates[weight]=weight_new

        positive_phase = theano.function(
            inputs=[n_iterations]+alphas,
            outputs=[],
            updates=updates
        )

        return positive_phase