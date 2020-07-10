
import numpy as np

def normalizeVec(vector):    
    den = np.sqrt(np.sum(vector**2))
    # print(f"den: {den}")
    normed_v = vector/den
    return np.round(normed_v,5)

def gramSchmidt(vectors):
    uVec_f = []
    numVecs = len(vectors)
    for idx, vector in enumerate(vectors):
        vecLength = len(vector)        
        if idx == 0:
            uVec = vector
        else:
            mu_sum = np.zeros(len(vector))
            for j in range(idx):
                mu = np.dot(uVec_f[vecLength*j:(vecLength*(j+1))],vector)
                mu = np.divide(mu, np.dot(uVec_f[vecLength*j:(vecLength*(j+1))],uVec_f[vecLength*j:(vecLength*(j+1))]))
                mu_sum += mu*uVec_f[vecLength*j:(vecLength*(j+1))]
            uVec = vector - mu_sum
        uVec_f = np.append(uVec_f,uVec)
    return np.reshape(uVec_f,[numVecs,vecLength])



# vectors = np.array([[2,1,1],[1,0,2],[1,0,0]])
vectors = np.array([[4,1,3,-1],[2,1,-3,4],[1,0,-2,7],[6,2,9,-5]])
print(f"Orthogonal Vectors: \n{gramSchmidt(vectors)}\n")


print("OrthoNormal Vectors:")
for vector in gramSchmidt(vectors):
    # print(f"Vector: \n{vector}")
    print( np.round(vector,5))
