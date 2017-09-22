from random import random
from random import seed
import numpy as np
import math



#### TD NOTES #####
#  alpha * (Pred_B - Pred_C)
#
#
##################

def gen_seq():
    values = "ABCDEFG"
    seq = [(values.index("D"), "D")]
    current = "D"
    while not current in ["A", "G"]:
        i = values.index(current)
        move = 1 if random() > 0.5 else -1
        current = values[i + move]
        i += move
        seq.append((i,current))
    return seq

def seq_to_x(seq, t = None):
    if t == None:
        t = len(seq)
    x_s = np.zeros((7, t), dtype='int')
    i = 0
    for i in range(0, t):
        step = seq[i]
        x_s[step[0],i] = 1
    return x_s
    

def lambda_weights(seq, lam, t = None):
    t = t+1 # make start at 1
    lambdas = [lam**(t-k) for k in range(1, t+1)]
    return np.array(lambdas)

ii = 0
def d_w(seq, a, lam, t, w):
    #print "\n\n>>> t=", t, len(seq)
    global ii
    ii += 1
    x = seq_to_x(seq)
    x_t = x[:, t]
    p_t = x[:,t].dot(w)

    p_t_plus_1 = None
    if (t >= len(seq) -1):
        p_t_plus_1 = 1 if seq[-1][1] == 'G' else 0
    else:
        p_t_plus_1 = x[:,t+1].dot(w)
        
    l = lambda_weights(seq, lam, t)
    #print "lam:",lam, "t:", t, "l:", l
    x_t = x[:, 0:t+1]
    l_t = l[0:t+1]
    #print x_t
    #print "l_t", l_t,t
    #print "x_t", x_t
    
    #print x_t * l_t
    summation = np.sum(x_t * l_t, axis=1)
    
    #print summation
    #print x[:,t]
    #print summation
    #print "sum:", summation
    #print "error:", (p_t_plus_1 - p_t)
    d_w_s = a * (p_t_plus_1 - p_t) * summation
    for ww in d_w_s:
        if math.isnan(ww):
            print ','.join([str(i) for i in d_w_s] ) 
            print "t:", str(t), a, summation, w
            print seq
            raise "Shit"
    return d_w_s


def start_probs():
    return np.array([random() for i in range(0, 7)])

def training_seqs(max):
    seq_s = []
    for i in range(0, max):
        seq_s.append(gen_seq())
    return seq_s

seed(5)
training = []
for i in range(1):
    training.append(training_seqs(10))

weights = np.ones(7) * .5
seq = gen_seq()
alpha = 0.1
lam = .5


def get_d_weights(seq, alpha, lam, weights):
    deltas = []
    for t in range(0, len(seq)):
        deltas.append(d_w(seq, alpha, lam, t, weights))
    return np.sum(deltas, axis=0)

weights = np.ones(7) * .5
seq = [(3, 'D'), (2, 'C'), (3, 'D'), (4, 'E'), (3, 'D'), (2, 'C'), (3, 'D'), (4, 'E'), (3, 'D'), (4, 'E'), (5, 'F'), (4, 'E'), (3, 'D'), (4, 'E'), (3, 'D'), (2, 'C'), (3, 'D'), (4, 'E'), (3, 'D'), (4, 'E'), (5, 'F'), (6, 'G')]
for i in range(3):
    d_weights =  get_d_weights(seq, 0.4, 0, weights)
    weights += d_weights
    print d_weights


def full_test(alpha=0.1, lam=0.5):
    weights = np.ones(7) * .5
    training = []
    for i in range(100):
        #training = [training_seqs(10)]
        training.append(training_seqs(10))
    i = 0
    for t in training:
        i+= 1
        matches = 0
        while matches < 1:
            deltas = []
            for seq in t:
                d_weights =  get_d_weights(seq, alpha, lam, weights)
                deltas.append(d_weights)
            d_weights = np.average(deltas, axis=0)
            done = True
            for w in d_weights:
                if abs(w) > .1:
                    done = False
            if done:
                matches+=1
            weights = weights + d_weights
    return weights


def simple():
    seq = gen_seq()
    print seq
    d_w = get_d_weights(seq, alpha, lam, weights)
    print weights + d_w
    
#simple()
#print full_test()

for alpha in [0.4]:#[0.0,0.1,0.2,0.3,0.4,0.5,0.6]:
    for lam in [0]: #,.8,.3,1]:
        weights = full_test(alpha, lam)
        weight_strs = [str(w) for w in weights.tolist()]
        print ','.join(["result", str(alpha), str(lam)] + weight_strs)


best = [0, 1/6.0, 1/3.0, 1/2.0, 2/3.0, 5/6.0, 1.0]
#print "iterations", i
#p
#
#for seq in training:
#    for i in range(len(seq)):
#        weights += d_w_0(seq,alpha, lam, i, weights)
#        print weights


# The prediction is based on weights and P(xt, w).
