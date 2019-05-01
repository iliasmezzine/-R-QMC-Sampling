import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sobol_seq as sob
from numpy.linalg import eigh

###############################################################################################################################
################################# ASIAN OPTION + GBM PATH GENERATION FUNCTIONS ################################################
###############################################################################################################################

def times(T,steps): #Equally-spaced list of the time interval [0,T]
    return [T*i/steps for i in range(1,steps+1)]

def covmat(t): #Returns the Covariance Matrix of a Brownian Path
    x = np.zeros([len(t),len(t)])
    for i in range(len(t)):
        for j in range(len(t)):
            x[i,j] = min(t[i],t[j])
    return np.array(x)

def PCA(cm): #computes the PCA*sqrt(D)
    pcadec = eigh(cm)

    #flip eigval/eigvect to descending order
    eigv = np.flip(pcadec[0],0)
    for x in range(pcadec[1].shape[0]):
        pcadec[1][x,:] = np.flip(pcadec[1][x,:],0)
    #compute Psqrt(D)
    return np.dot(pcadec[1],np.diag(np.sqrt(eigv)))


def chol(M): #Returns the Cholesky Factorization of any Definite Positive Matrix
    try :
        return np.array(np.linalg.cholesky(M))
    except:
        return "Non DP-Matrix"

def brw_path(T,steps): # Generates a Brownian path with steps [t[0],...,t[-1]] using standard discretization (Euler)
    chol_dec = chol(covmat(times(T,steps)))
    path = np.dot(chol_dec,np.random.normal(0,1,steps))
    return path


def gbm_path(s0,T,steps,r,v): # Generates a Geometric brownian path with steps [t[0],...,t[-1]] and chosen vol, drift
    bm = brw_path(T,steps)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def asian(s0,T,steps,r,v,K): # Returns a sample Asian Call price
    return np.exp(-r*T)*max(np.mean(gbm_path(s0,T,steps,r,v))-K,0)

def asianMC(s0,T,steps,r,v,K,nSim): #MC Simulation for the above function
    return np.mean([asian(s0,T,steps,r,v,K) for i in range(nSim)])

def bm_path_drifted(drift,T,steps): #Returns a drifted brownian path with time indexes in T
    
    path = brw_path(T,steps) 
    drifted_path = [path[i]+times(T,steps)[i]*drift for i in range(len(path))]  
    return np.array(drifted_path)

def gbm_path_drifted(drift,s0,T,steps,r,v): #Returns a drifted GBM path with indexes in T
    bm = bm_path_drifted(drift,T,steps)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def asianGirsanov(drift,s0,T,steps,r,v,K): #Returns a drifted Asian Call option Price Sample
    
    path = brw_path(T,steps)
    
    mult_drift = np.exp(-drift*path[-1] - 0.5*(drift**2)*times(T,steps)[-1])
    pathD = [path[i]+drift*times(T,steps)[i] for i in range(len(path))]
    gbmD = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*pathD[i]) for i in range(len(pathD))]
    
    opt_price = np.exp(-r*T)*mult_drift*max(np.mean(gbmD)-K,0)

    return opt_price

######################################################################################################################
##################### LOW DISCREPANCY AND BROWNIAN BRIDGE FUNCTIONS ##################################################
######################################################################################################################

def sobol(dim,seed): # Generates the seed-th term of the p-dimensional Sobol Sequence (WARNING P MUST BE <=40)
    return list(sob.i4_sobol(dim,seed)[0])
    
def phi(x,y): #Returns the mid-integer index between two indexes
    if int((x-y)/2) == 0:
        return 0
    else:
        return y + int((x-y)/2) 
  
def alternate(l_max,l_min): #Returns an alternated list from two merged lists (used for BB generation)
    z = [l_max[0]]
    for i in range(len(l_min)):
        z = z + [l_min[i]] + [l_max[i+1]]
    return z

def discretize(lst): #Adds one discretization step to the Brownian Bridge
    h=[]
    nxt = [phi(lst[i+1],lst[i]) for i in range(len(lst)-1)]
    for i in range(len(alternate(lst,nxt))):
        if alternate(lst,nxt)[i] != 0 or i==0 :
            h+=[alternate(lst,nxt)[i]]
    return h

def ninv(x): #Returns the inverse cumulative standard function of some point x in ]0,1[
    return norm.ppf(x)

def generate_regular_bridge(T): #Generate a regular (pseudorandom) BB sample from the Times Index T

    b_start = np.random.normal(0,1,1)*np.sqrt(T[0])
    b_end = b_start + np.random.normal(0,1,1)*np.sqrt(T[1]-T[0])

    G = [[0,len(T)-1],[b_start,b_end]]
    
    while len(G[0]) != len(T):
        new_brownian = []
        for k in range(len(G[0])-1):
            if phi(G[0][k+1],G[0][k]) == 0:
                new_brownian = new_brownian + [0]
            else:
                v = T[phi(G[0][k+1], G[0][k])]

                a = G[1][k]
                b = G[1][k+1] 

                w = T[G[0][k+1]] 
                u = T[G[0][k]]
                dt = w-u
                mu = a*(w-v)/dt + b*(v-u)/dt
                var = (v-u)*(w-v)/dt

                x = np.random.normal(mu,np.sqrt(var),1)[0]
                new_brownian = new_brownian + [x]

        G[0] = discretize(G[0])
        G[1] = [x for x in alternate(G[1],new_brownian) if x != 0]
    return G[1]

def generate_sobol_bridge(T,seed): #Generate a Sobol-Based BB sample from the Times Index T at given seed
    sample = sobol(len(T),seed)
    
    b_start = ninv(sample[1])*np.sqrt(T[0])
    b_end = b_start + ninv(sample[0])*np.sqrt(T[1]-T[0])

    del sample[0:2]

    G = [[0,len(T)-1],[b_start,b_end]] #Gauche : indices déjà générés, droite, valeur du BM à ces indices
    while len(G[0]) != len(T):
        new_brownian = []
        for k in range(len(G[0])-1):
            
            if phi(G[0][k+1],G[0][k]) == 0: #S'il n'existe aucun indice intermédiaire, ajouter 0 aux nouvelles valeurs du BM 
                new_brownian = new_brownian + [0]
            else:
                
                v = T[phi(G[0][k+1], G[0][k])]
                a = G[1][k]
                b = G[1][k+1] 
                w = T[G[0][k+1]] 
                u = T[G[0][k]]
                dt = w-u
                mu = a*(w-v)/dt + b*(v-u)/dt
                var = (v-u)*(w-v)/dt
                x = mu + np.sqrt(var)*ninv(sample[0])
                del sample[0]
                new_brownian = new_brownian + [x]
                
        G[0] = discretize(G[0])
        G[1] = [x for x in alternate(G[1],new_brownian) if x != 0]
        
    return G[1]

def generate_sobol_PCA(T,seed): #Sobol-Based BM Path with PCA sample from the Times Index T at given seed
    return np.dot(PCA(covmat(T)),ninv(sobol(len(T),seed)))
    

def GBM_regular(s0,T,steps,r,v): # Random Geometric brownian path with steps [t[0],...,t[-1]] and chosen vol, drift
    bm = generate_regular_bridge(times(T,steps))
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def GBM_sobol(s0,T,steps,r,v,seed): # GBM with Brownian Bridge on a given sobol coordinate (seed)
    bm = generate_sobol_bridge(times(T,steps),seed)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def GBM_sobol_PCA(s0,T,steps,r,v,seed): # GBM with PCA on a given sobol coordinate (seed)
    bm = generate_sobol_PCA(times(T,steps),seed)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def asianRegular(s0,T,steps,r,v,K): # Returns a sample Asian Call price with Brownian Bridge Transformation
    return np.exp(-r*T)*max(np.mean(GBM_regular(s0,T,steps,r,v))-K,0)

def asianSobol(s0,T,steps,r,v,K,seed): # Returns a sample Asian Call price with Sobol + Brownian Bridge Transformation (seed = index of sobol seq)

    return np.exp(-r*T)*max(np.mean(GBM_sobol(s0,T,steps,r,v,seed))-K,0)

def asianSobolPCA(s0,T,steps,r,v,K,seed): # Returns a sample Asian Call price with Sobol +PCA Transformation (seed = index of sobol seq)

    return np.exp(-r*T)*max(np.mean(GBM_sobol_PCA(s0,T,steps,r,v,seed))-K,0)


###########################################################################
########################### OPTION PARAMETERS #############################
###########################################################################


def means_of(x):
    return [np.mean(x[:i]) for i in range(len(x))]

def asianQMCprice(s0,T,steps,r,v,K,nSim):
    return np.mean([asianSobol(s0,T,steps,r,v,K,300+i) for i in range(nSim)])

def asianMCprice(s0,T,steps,r,v,K,nSim):
    return np.mean([asianRegular(s0,T,steps,r,v,K) for i in range(nSim)])

def compare_asians(s0,T,steps,r,v,K,nSim):
    
    tag = "s = {}, T = {} , steps = {} , r = {} , v = {} , K = {} ".format(s0,T,steps,r,v,K)
    
    x = [asianSobol(s0,T,steps,r,v,K,300+i) for i in range(nSim)]
    y = [asianRegular(s0,T,steps,r,v,K) for i in range(nSim)]
    
    plt.plot(means_of(x),dashes=[1,1], label="QMCSobol" + tag)
    plt.plot(means_of(y),dashes=[4,1], label="MC" + tag)
    
    plt.legend()
    plt.show()

#######################################################################
############################## DISTANCE MATRICES ######################
#######################################################################

def m_infty_asian(max_step,disc_step,max_strike,disc_strike,nSim,s0,T,r,v): #Returns the matrix of asian prices with step and strike discretization
    
    strikes = np.linspace(0,max_strike,disc_strike)
    steps= np.array([int(x) for x in np.linspace(0,max_step,disc_step) if x != 0])

    output = np.zeros([len(strikes),len(steps)])
    for i in range(len(strikes)):
       for j in range(len(steps)):
            output[i,j] = asianQMCprice(s0,T,steps[j],r,v,strikes[i],nSim)
    return output

def distAsianQMC(max_step,disc_step,max_strike,disc_strike,nSim,s0,T,r,v):
    #Pour chaque strike et disc_step, on génère d'abord nSim échantillons, ensuite on remplit toutes les matrices puis on calcule la distance
    strikes = np.linspace(0,max_strike,disc_strike)
    steps= np.array([int(x) for x in np.linspace(0,max_step,disc_step) if x != 0])
    
    sims_Sobol = np.ndarray([len(strikes),len(steps),nSim])
    sims_MC = np.ndarray([len(strikes),len(steps),nSim])
    
    #First, fill the sims matrix with mean prices
    for i in range(len(strikes)):
       for j in range(len(steps)):
           means_Sobol = means_of([asianSobol(s0,T,steps[j],r,v,strikes[i],300+n) for n in range(nSim)])
           means_MC = means_of([asianRegular(s0,T,steps[j],r,v,strikes[j]) for n in range(nSim)])
           for k in range(nSim):
               sims_Sobol[i,j,k] = means_Sobol[k]
               sims_MC[i,j,k] = means_MC[k]
               
    #Then, return the distance matrix
    dist_Sobol = [np.sqrt(np.sum((sims_Sobol[:,:,i]-sims_Sobol[:,:,-1])**2)) for i in range(nSim)] #distance with Sobol & BB
    dist_MC = [np.sqrt(np.sum((sims_MC[:,:,i]-sims_MC[:,:,-1])**2)) for i in range(nSim)] #distance with BB

    plt.plot(dist_Sobol, dashes=[1,1], label="Asian + Brownian Bridge & Sobol ")
    plt.plot(dist_MC, dashes=[3,1] ,label="Asian + Drift Change")
    plt.legend()
    plt.show() 
    return [dist_Sobol , dist_MC]

def distAsianQMCPCA(max_step,disc_step,max_strike,disc_strike,nSim,s0,T,r,v):
    #Pour chaque strike et disc_step, on génère d'abord nSim échantillons, ensuite on remplit toutes les matrices puis on calcule la distance
    strikes = np.linspace(0,max_strike,disc_strike)
    steps= np.array([int(x) for x in np.linspace(0,max_step,disc_step) if x != 0])
    
    sims_SobolBB = np.ndarray([len(strikes),len(steps),nSim])
    sims_SobolPCA = np.ndarray([len(strikes),len(steps),nSim])
    sims_MC = np.ndarray([len(strikes),len(steps),nSim])
    
    #First, fill the sims matrix with mean prices
    for i in range(len(strikes)):
        print("strike " + str(strikes[i]))
        for j in range(len(steps)):
           print("steps" + str(steps[j]))
           means_SobolPCA = means_of([asianSobolPCA(s0,T,steps[j],r,v,strikes[i],300+n) for n in range(nSim)])
           means_SobolBB = means_of([asianSobol(s0,T,steps[j],r,v,strikes[i],300+n) for n in range(nSim)])
           means_MC = means_of([asianRegular(s0,T,steps[j],r,v,strikes[i]) for n in range(nSim)])
           for k in range(nSim):
               sims_SobolBB[i,j,k] = means_SobolBB[k]
               sims_SobolPCA[i,j,k] = means_SobolPCA[k]
               sims_MC[i,j,k] = means_MC[k]
               
    #Then, return the distance matrix
    dist_SobolBB = [np.sqrt(np.sum((sims_SobolBB[:,:,i]-sims_SobolBB[:,:,-1])**2)) for i in range(nSim)] #distance with Sobol & BB
    dist_SobolPCA = [np.sqrt(np.sum((sims_SobolPCA[:,:,i]-sims_SobolPCA[:,:,-1])**2)) for i in range(nSim)] #distance with Sobol & BB
    dist_MC = [np.sqrt(np.sum((sims_MC[:,:,i]-sims_MC[:,:,-1])**2)) for i in range(nSim)] #distance with BB

    plt.plot(dist_SobolBB, label="Asian + BB & Sobol ",color="black")
    plt.plot(dist_SobolPCA, label="Asian + PCA & Sobol ",color="green")
    plt.plot(dist_MC,label="Asian + Drift Change",color="blue")
    plt.legend()
    plt.show()
    
    return [dist_SobolBB ,dist_SobolPCA, dist_MC]


#############################################################
################### RQMC FUNCTIONS ##########################
#############################################################


def rndunif(dim):
    return np.random.uniform(0,1,dim)

def fb(x): #float to binary floating point list. precision is kept from :20 bin. floating points
    binl = []
    s=x
    for i in range(20):
        binl.append(int(2*s))
        s = 2*s - int(2*s)
    return np.array(binl)

def rev(listx): #reverse function of fb
    x = np.array([2**-(i+1) for i in range(len(listx))])
    return np.dot(x,listx)
def xor1(x,y):
    if x!=y:
        return 1
    return 0
def xorlist(lst1,lst2): #XOR a list bitwise
    return np.array([xor1(lst1[i],lst2[i]) for i in range(len(lst1))])

def fXOR(x,y):#XOR bitwise 2 floats 
    return rev(xorlist(fb(x),fb(y)))

vXOR2 = np.vectorize(fXOR)

#RQMC - SOBOL : generate separate sobol batches and digitally shift each batch with a random uniform vector

def Shift_CPR(dim,batch_size,nsamples): 
    batch = [] 
    for n in range(nsamples):
        U = rndunif(dim)
        batch+=[[vXOR2(sobol(dim,s),U) for s in range(n*batch_size,(n+1)*batch_size)]] 
        print("done shifting batch : " + str(n+1))
    return batch

def dShift(dim,batch_size,nsamples): 
    batch = []
    unshifted = [sobol(dim,s+500) for s in range(batch_size,2*batch_size)]
    for n in range(nsamples):
        U = rndunif(dim)
        batch+=[[vXOR2(S,U) for S in unshifted]] 
        print("done shifting batch : " + str(n+1))
    return batch

def S_bridge(T,S): #Generate a Sobol-Based BB from any sample
    sample = S
    b_start = ninv(sample[1])*np.sqrt(T[0])
    b_end = b_start + ninv(sample[0])*np.sqrt(T[1]-T[0])

    del sample[0:2]

    G = [[0,len(T)-1],[b_start,b_end]] #Gauche : indices déjà générés, droite, valeur du BM à ces indices
    while len(G[0]) != len(T):
        new_brownian = []
        for k in range(len(G[0])-1):
            
            if phi(G[0][k+1],G[0][k]) == 0: #S'il n'existe aucun indice intermédiaire, ajouter 0 aux nouvelles valeurs du BM 
                new_brownian = new_brownian + [0]
            else:
                
                v = T[phi(G[0][k+1], G[0][k])]
                a = G[1][k]
                b = G[1][k+1] 
                w = T[G[0][k+1]] 
                u = T[G[0][k]]
                dt = w-u
                mu = a*(w-v)/dt + b*(v-u)/dt
                var = (v-u)*(w-v)/dt
                x = mu + np.sqrt(var)*ninv(sample[0])
                del sample[0]
                new_brownian = new_brownian + [x]
                
        G[0] = discretize(G[0])
        G[1] = [x for x in alternate(G[1],new_brownian) if x != 0]

    return G[1]

def S_GBM(s0,T,steps,r,v,S): # GBM with Brownian Bridge on a given Sample
    bm = S_bridge(times(T,steps),S)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)


def S_GBM_Chol(s0,T,steps,r,v,S): # GBM with Cholesky dec on a given Sample
    bm = np.dot(chol(covmat(times(T,steps))),ninv(S))
    return np.array([s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))])

def S_GBM_PCA(s0,T,steps,r,v,S): # GBM with Cholesky dec on a given Sample
    bm = np.dot(PCA(covmat(times(T,steps))),ninv(S))
    return np.array([s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))])

def S_asian(x,K):
    return np.max(np.mean(x)-K,0)

s0 = 100
T = 1
K = 100
steps = 40
r = 0.05
v = 0.3

batch_size = 1000
nbatch = 10
nSim = batch_size*nbatch
x = dShift(steps,batch_size,nbatch)


asiansRQMC = [np.mean([S_asian(S_GBM_Chol(s0,T,steps,r,v,list(u)),K) for u in y]) for y in x]
#asiansMC = means_of([asianRegular(s0,T,steps,r,v,K) for i in range(nSim)])

#del asiansMC[0:2]

print(np.std(asiansRQMC))
#print(np.std(asiansMC))


