import numpy as np
import pandas as pd

N=64
cp=16
m=2000
snrdb=np.arange(0,21,2)

def mod(b):
    b=b.reshape(-1,2)
    s=(2*b[:,0]-1)+1j*(2*b[:,1]-1)
    return s/np.sqrt(2)

def lab(s):
    l=[]
    for i in s:
        if np.real(i)>0 and np.imag(i)>0:
            l.append(0)
        elif np.real(i)<0 and np.imag(i)>0:
            l.append(1)
        elif np.real(i)<0 and np.imag(i)<0:
            l.append(2)
        else:
            l.append(3)
    return np.array(l)

X=[]
Y=[]

for snr in snrdb:
    for _ in range(m):

        b=np.random.randint(0,2,N*2)
        x=mod(b)
        xt=np.fft.ifft(x)
        xcp=np.concatenate([xt[-cp:],xt])

        p=np.mean(abs(xcp)**2)
        snr_lin=10**(snr/10)
        nvar=p/snr_lin

        n=np.sqrt(nvar/2)*(np.random.randn(len(xcp))+1j*np.random.randn(len(xcp)))
        y=xcp+n

        y=y[cp:]
        yf=np.fft.fft(y)

        l=lab(x)

        for i in range(N):
            X.append([np.real(yf[i]),np.imag(yf[i]),snr])
            Y.append(l[i])

X=np.array(X)
Y=np.array(Y)

df=pd.DataFrame(X,columns=["Re","Im","SNR"])
df["Label"]=Y

df.to_csv("data/dataset.csv",index=False)

print("Dataset shape:",df.shape)
print("Saved to data/dataset.csv")
