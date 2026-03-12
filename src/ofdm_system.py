import numpy as np
import matplotlib.pyplot as plt

N=64
cp=16
m=1000
snrdb=np.arange(0,21,2)
ber=[]

def qpsk_mod(b):
    b=b.reshape(-1,2)
    s=(2*b[:,0]-1)+1j*(2*b[:,1]-1)
    return s/np.sqrt(2)

def qpsk_demod(s):
    b=np.zeros((len(s),2))
    b[:,0]=np.real(s)>0
    b[:,1]=np.imag(s)>0
    return b.reshape(-1)

for snr in snrdb:

    e=0
    t=0

    for _ in range(m):

        b=np.random.randint(0,2,N*2)

        x=qpsk_mod(b)

        xt=np.fft.ifft(x)

        xcp=np.concatenate([xt[-cp:],xt])

        p=np.mean(abs(xcp)**2)

        snr_lin=10**(snr/10)

        nvar=p/snr_lin

        n=np.sqrt(nvar/2)*(np.random.randn(len(xcp))+1j*np.random.randn(len(xcp)))

        y=xcp+n

        y=y[cp:]

        yf=np.fft.fft(y)

        br=qpsk_demod(yf)

        e+=np.sum(b!=br)
        t+=len(b)

    ber.append(e/t)

plt.semilogy(snrdb,ber,'o-')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("OFDM BER vs SNR")
plt.grid()
plt.show()
