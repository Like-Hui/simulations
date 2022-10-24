from scipy.special import erfc
from numpy import (
    arcsin as asin, sin, exp, pi, sqrt, inf,
    array, matrix, concatenate, vectorize
)

bivariate_normal_cdf = lambda h, k, r: bvnu(-h, -k, r)

phid = lambda z: erfc(-z/sqrt(2))/2

def bvnu_(h, k, r):
    """
        Computes bivariate normal probabilities
        X > h and Y > k, where (X, Y) are standard normals
        with correlation coefficient r
        Parameters:
            h: 1st lower integration limit
            k: 2nd lower integration limit
            r: correlation coefficient
        Example:
            p = bivariate_normal_cdf(3, 1, .35)
        Note: to compute the probability that X < h and Y < k, 
        use bvnu(-h, -k, r)
    """

    if (h ==  inf) or (k ==  inf): 
        return 0
    elif h == -inf: 
        return 1 if k == -inf else phid(-k)
    elif k == -inf: 
        return phid(-h)
    elif r == 0: 
        return phid(-h)*phid(-k)

    tp, hk, bvn = 2*pi, h*k, 0

    if abs(r) < 0.3:
        w = [0.1713244923791705, 0.3607615730481384, 0.4679139345726904]
        x = [0.9324695142031522, 0.6612093864662647, 0.2386191860831970]
    elif abs(r) < 0.75:
        w = [.04717533638651177, 0.1069393259953183, 0.1600783285433464]
        w+= [0.2031674267230659, 0.2334925365383547, 0.2491470458134029]
        x = [0.9815606342467191, 0.9041172563704750, 0.7699026741943050]
        x+= [0.5873179542866171, 0.3678314989981802, 0.1252334085114692]
    else:
        w = [.01761400713915212, .04060142980038694, .06267204833410906]
        w+= [.08327674157670475, 0.1019301198172404, 0.1181945319615184]
        w+= [0.1316886384491766, 0.1420961093183821, 0.1491729864726037]
        w+= [0.1527533871307259]
        x = [0.9931285991850949, 0.9639719272779138, 0.9122344282513259]
        x+= [0.8391169718222188, 0.7463319064601508, 0.6360536807265150]
        x+= [0.5108670019508271, 0.3737060887154196, 0.2277858511416451]
        x+= [0.07652652113349733]

    w = matrix(w+w)
    x = concatenate([1-array(x),1+array(x)]); 

    if abs(r) < 0.925:
        hs = ( h*h + k*k )/2
        asr = asin(r)/2;  
        sn = sin(asr*x); 
        bvn = exp((sn*hk-hs)/(1-sn**2)) * w.H
        bvn = bvn*asr/tp + phid(-h)*phid(-k); 
    elif r < 0:
        k, hk = -k, -hk
        if abs(r) < 1:
            as_ = 1-r**2; 
            a = sqrt(as_); 
            bs = (h-k)**2;
            asr = -( bs/as_ + hk )/2
            c = (4-hk)/8 
            d = (12-hk)/80
            if asr > -100: 
                bvn = a*exp(asr)*(1-c*(bs-as_)*(1-d*bs)/3+c*d*as_**2)
            if hk  > -100: 
                b = sqrt(bs)
                sp = sqrt(tp)*phid(-b/a)
                bvn = bvn - exp(-hk/2)*sp*b*( 1 - c*bs*(1-d*bs)/3 )
            a = a/2
            xs = (a*x)**2
            asr = -(bs/xs + hk)/2; 
            xs = xs[asr > -100]
            sp = ( 1 + c*xs*(1+5*d*xs) )
            rs = sqrt(1-xs); 
            ep = exp( -(hk/2)*xs/(1+rs)**2 )/rs; 
            bvn = ( a*( (exp(asr(ix))*(sp-ep))*w(ix).H ) - bvn )/tp;

        if r > 0: 
            bvn =  bvn + phid(-max(h, k))
        elif h >= k: 
            bvn = -bvn
        else:
            L = (phid(k)-phid(h)) if h < 0 else (phid(-h)-phid(-k))
            bvn =  L - bvn;
    
    p = max(0, min(1, bvn));

    return p.item()

bvnu = vectorize(bvnu_)


if __name__ == "__main__":
    import numpy as np
    rho = 0.2
    n_samples = 100000
    X1,X2 = np.split(np.random.multivariate_normal(
        np.zeros(2), np.array([[1,rho],[rho,1]]), n_samples
    ),2,axis=1)
    count = np.count_nonzero((X1<0)&(X2<0))/len(X1)
    print(count, bivariate_normal_cdf(0,0,rho))