from model import Network
import numpy as np
import sys
from sys import stdout
import time

def train_net(net):

    path         = net.path
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    n_epochs     = net.hyperparameters["n_epochs"]
    batch_size   = net.hyperparameters["batch_size"]
    n_it_neg     = net.hyperparameters["n_it_neg"]
    n_it_pos     = net.hyperparameters["n_it_pos"]
    epsilon      = net.hyperparameters["epsilon"]
    beta         = net.hyperparameters["beta"]
    alphas       = net.hyperparameters["alphas"]

    print "name = %s" % (path)
    print "architecture = 784-"+"-".join([str(n) for n in hidden_sizes])+"-10"
    print "number of epochs = %i" % (n_epochs)
    print "batch_size = %i" % (batch_size)
    print "n_it_neg = %i"   % (n_it_neg)
    print "n_it_pos = %i"   % (n_it_pos)
    print "epsilon = %.1f" % (epsilon)
    print "beta = %.1f" % (beta)
    print "learning rates: "+" ".join(["alpha_W%i=%.3f" % (k+1,alpha) for k,alpha in enumerate(alphas)])+"\n"

    n_batches_train = 50000 / batch_size
    n_batches_valid = 10000 / batch_size

    start_time = time.clock()

    for epoch in range(n_epochs):

        ### TRAINING ###

        # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
        measures_sum = [0.,0.,0.]
        gW = [0.] * len(alphas)

        for index in xrange(n_batches_train):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(index)

            # FREE PHASE
            net.free_phase(n_it_neg, epsilon)

            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            stdout.write("\r%2i-train-%5i E=%.1f C=%.5f error=%.3f%%" % (epoch, (index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
            stdout.flush()

            # WEAKLY CLAMPED PHASE
            sign = 2*np.random.randint(0,2)-1 # random sign +1 or -1
            beta = np.float32(sign*beta) # choose the sign of beta at random

            Delta_logW = net.weakly_clamped_phase(n_it_pos, epsilon, beta, *alphas)
            gW = [gW1 + Delta_logW1 for gW1,Delta_logW1 in zip(gW,Delta_logW)]

        stdout.write("\n")
        dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
        print "   "+" ".join(["dlogW%i=%.3f%%" % (k+1,dlogW1) for k,dlogW1 in enumerate(dlogW)])

        net.training_curves["training error"].append(measures_avg[-1])

        ### VALIDATION ###
        
        # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR
        measures_sum = [0.,0.,0.]

        for index in xrange(n_batches_valid):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(n_batches_train+index)

            # FREE PHASE
            net.free_phase(n_it_neg, epsilon)
            
            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.2f%%" % ((index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
            stdout.flush()

        stdout.write("\n")

        net.training_curves["validation error"].append(measures_avg[-1])

        duration = (time.clock() - start_time) / 60.
        print("   duration=%.1f min" % (duration))

        # SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
        net.save_params()


# HYPERPARAMETERS FOR A NETWORK WITH 1 HIDDEN LAYER
net1 = "net1", {
"hidden_sizes" : [500],
"n_epochs"     : 25,
"batch_size"   : 20,
"n_it_neg"     : 20,
"n_it_pos"     : 4,
"epsilon"      : np.float32(.5),
"beta"         : np.float32(.5),
"alphas"       : [np.float32(.1), np.float32(.05)]
}

# HYPERPARAMETERS FOR A NETWORK WITH 2 HIDDEN LAYERS
net2 = "net2", {
"hidden_sizes" : [500,500],
"n_epochs"     : 60,
"batch_size"   : 20,
"n_it_neg"     : 150,
"n_it_pos"     : 6,
"epsilon"      : np.float32(.5),
"beta"         : np.float32(1.),
"alphas"       : [np.float32(.4), np.float32(.1), np.float32(.01)]
}

# HYPERPARAMETERS FOR A NETWORK WITH 3 HIDDEN LAYERS
net3 = "net3", {
"hidden_sizes" : [500,500,500],
"n_epochs"     : 500,
"batch_size"   : 20,
"n_it_neg"     : 500,
"n_it_pos"     : 8,
"epsilon"      : np.float32(.5),
"beta"         : np.float32(1.),
"alphas"       : [np.float32(.128), np.float32(.032), np.float32(.008), np.float32(.002)]
}

if __name__ == "__main__":

    # TRAIN A NETWORK WITH 1 HIDDEN LAYER
    train_net(Network(*net1))