import cPickle
from external_world import External_World
import numpy as np
import os
import theano
import theano.tensor as T

def rho(s):
    return T.clip(s,0.,1.)
    #return T.nnet.sigmoid(4.*s-2.)

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
        self.free_phase              = self.__build_free_phase()
        self.weakly_clamped_phase    = self.__build_weakly_clamped_phase()

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
        squared_norm    =   sum( [T.batched_dot(rho(layer),rho(layer))       for layer      in layers] ) / 2.
        linear_terms    = - sum( [T.dot(rho(layer),b)                        for layer,b    in zip(layers,self.biases)] )
        quadratic_terms = - sum( [T.batched_dot(T.dot(rho(pre),W),rho(post)) for pre,W,post in zip(layers[:-1],self.weights,layers[1:])] )
        return squared_norm + linear_terms + quadratic_terms

    # COST FUNCTION, DENOTED BY C
    def __cost(self, layers):
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(axis=1)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_energy(self, layers, beta):
        return self.__energy(layers) + beta * self.__cost(layers)

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

    def __build_free_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__energy(layers))
            layers_dot = T.grad(-E_sum, list(layers)) # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles,self.layers[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)

        free_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )

        return free_phase

    def __build_weakly_clamped_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        alphas = [T.fscalar("alpha_W"+str(r+1)) for r in range(len(self.weights))]

        def step(*layers):
            F_sum = T.sum(self.__total_energy(layers, beta))
            layers_dot = T.grad(-F_sum, list(layers)) # temporal derivative of the state (weakly clamped trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_weakly_clamped = [layer[-1] for layer in layers]

        E_mean_free           = T.mean(self.__energy(self.layers))
        E_mean_weakly_clamped = T.mean(self.__energy(layers_weakly_clamped))
        biases_dot            = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.biases,  consider_constant=layers_weakly_clamped)
        weights_dot           = T.grad( (E_mean_weakly_clamped-E_mean_free) / beta, self.weights, consider_constant=layers_weakly_clamped)

        biases_new  = [b - alpha * dot for b,alpha,dot in zip(self.biases[1:],alphas,biases_dot[1:])]
        weights_new = [W - alpha * dot for W,alpha,dot in zip(self.weights,   alphas,weights_dot)]
        
        Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights,weights_new)]
        
        for bias, bias_new in zip(self.biases[1:],biases_new):
            updates[bias]=bias_new
        for weight, weight_new in zip(self.weights,weights_new):
            updates[weight]=weight_new

        weakly_clamped_phase = theano.function(
            inputs=[n_iterations, epsilon, beta]+alphas,
            outputs=Delta_log,
            updates=updates
        )

        return weakly_clamped_phase
