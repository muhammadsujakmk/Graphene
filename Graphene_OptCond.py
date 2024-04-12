import numpy as np
import mpmath as mp
from mpmath import log, exp
from mpmath import cosh, sinh
from mpmath import quad 
import scipy.integrate as integrate
import matplotlib.pyplot as plt



## Universal Constant - SI (MKS)
ee = 1.60217663e-19 #elementary charge
pi = np.pi
eps0 = 8.8541878128e-12 # Vacuum permittivity, F/m
hh = 6.62607015e-34 # Planck constant, J. s
hheV = 4.135667696e-15 # Planck constant, eV. s
hbar = hh/2/pi      # Reduced Planck constant (J. s)
hbar_ = 1.05457182e-34 
hbareV = hheV/2/pi      # Reduced Planck constant (eV. s)
c = 299792458       # speed of light, m/s
kB = 1.380649e-23 # Boltzmann constant, J/K
kBeV = 8.617333262e-5 # Boltzmann constant, eV/K
vf = 1.1e6 # Fermi velocity of electron, m/s
X = ee**2/hbar/c 


## Unit adjustment
nm = 1e-9
um = 1e-6

## Parameter used here from Emani,Nanoletter,2012
#nd = np.array([3e11,1e12,1e13])
nd_ = [6e16] # Carrier densities, m^-2
T = 300 
t_Gr = 1*nm # Graphene thickness

# intraband 
def sigma_intra(wavelength,omega_T,omega_F,tau,ee,hbar):
    omega = 2*np.pi*c/wavelength
    sig_intra = 2 * ee**2 * omega_T * 1j
    sig_intra /= (pi * hbar * (omega+1j/tau))
    sig_intra *= log(2*cosh(omega_F/2/omega_T))
    return complex(sig_intra)

# interband
def sigma_inter(wavelength,omega_T,omega_F,c,ee,hbar):
    omega = 2*np.pi*c/wavelength 
    omegaX = omega*X
    omega_TX = omega_T*X 
    omega_FX = omega_F*X
    Hw = lambda w: sinh(w/omega_TX)/(cosh(omega_FX/omega_TX)+cosh(w/omega_TX))
    
    intg = quad(lambda w: ( Hw(w/2)-Hw(omegaX/2) ) / (omegaX**2 - w**2), [0,np.inf])*X
    return complex(ee**2/4/hbar*(Hw(omegaX/2)+1j*2*omega/pi*intg))


epsReal,epsImag = [],[]
sigReal,sigImag = [],[]
for nd in nd_:
    epsReal.clear()
    sigReal.clear()
    epsImag.clear()
    sigImag.clear()
    fileOut = open(f'Graphene_OptCond_CarrierDensity={nd*1e-4:.1e} cm^-1.txt','w')
    fileOut.write("##Wavelength (um) | epsReal | epsImag | sigmaReal | sigmaImag | sigma\n")
    wavelength =np.arange(.1,10,1)*um
    #Ef = hbar * vf * (pi * nd)**.5
    EfeV = .24# hbareV * vf * (pi * nd)**.5
    Ef = EfeV*ee 
    mu = 1#tau * ee * vf**2 / Ef
    tau = mu*Ef/ee/vf**2 
    print(tau)
    omega_T = kB*T/hbar
    omega_F = EfeV/hbareV
    rat = omega_T/omega_F
    print('\u03c9_T/\u03c9_{F}'+f'={rat} Carrier Mobiliy = {mu*1e4:.0f} Fermi Energy = {EfeV:.3f} eV')
    for wv in wavelength:
        omega = 2*np.pi*c/wv
        sig_intra = sigma_intra(wv,omega_T,omega_F,tau,ee,hbar)
        sig_inter = sigma_inter(wv,omega_T,omega_F,c,ee,hbar)
        sigma = sig_intra+sig_inter

        eps_Gr = 1j*sigma/omega/eps0/t_Gr
        epsReal.append(eps_Gr.real) 
        epsImag.append(eps_Gr.imag)
        sigReal.append(sigma.real) 
        sigImag.append(sigma.imag) 
        res = f"{hbar*omega/Ef} {eps_Gr.real} {eps_Gr.imag} {eps_Gr} {sigma.real*4*hbar/ee**2} {sigma.imag*4*hbar/ee**2} {sigma*4*hbar/ee**2}\n" 
        #res = f"{wv/um} {sigma.real} {sigma.imag} {sigma}\n" 
        fileOut.write(res)
    
    """ 
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_title(f'Carrier Density = {nd*1e-4:.0e} $cm^{{-2}}$')
    ax1.plot(wavelength/um,epsreal,linewidth=5)
    ax1.set_xlim(0,12)
    ax1.set_xlabel('Wavelength (\u00B5m)',fontsize=15)
    ax1.set_ylabel('Re(${\u03B5}$)',fontsize=15)

    ax2.plot(wavelength/um,Qimag,'--',linewidth=5)
    ax2.set_ylim(-4,42)
    ax2.set_ylabel('Im(${\u03B5}$)',fontsize=15)
    plt.show()
    """

fileOut.close()

### Plot graph
import pandas as pd
fig,ax1 = plt.subplots()
#ax2 = ax1.twinx()
c_ = ['k','r','b']
for i,nd in enumerate(nd_):
    fname = f'Graphene_OptCond_CarrierDensity={nd*1e-4:.1e} cm^-1.txt'
    df = pd.read_csv(fname,sep=" ")
    wvl = df[df.columns[0]]
    epsReal = df[df.columns[1]]
    epsImag = df[df.columns[2]]
    sigReal = df[df.columns[4]]
    sigImag = df[df.columns[5]]
    
    """    
    ax1.plot(wvl,epsReal,label='$n_{{2D}}$'+f' = {nd*1e-4:.0e} cm$^{({-2})}$',linewidth=5,color=c_[i])
    ax2.plot(wvl,epsImag,'--',linewidth=5,color=c_[i])
    """ 
    #ax1.plot(wvl,sigReal,label='$n_{{2D}}$'+f'={nd*1e-4:.0e} cm$^{({-2})}$',linewidth=5,color=c_[i])
    ax1.plot(wvl,sigReal,label='Real(\u03c3)',linewidth=5,color='black')
    ax1.plot(wvl,sigImag,'--',label='Imag(\u03c3)',linewidth=5,color='red')

ax1.set_xlim(0,6)
ax1.tick_params(axis='both',labelsize=15)
#ax1.set_xlabel('Wavelength (\u00B5m)',fontsize=15)
ax1.set_xlabel('$\u0127\u03c9/E_{F}$ ',fontsize=20)
#ax1.set_ylabel('$\u03c3\u00D7e^{2}/4\u0127$',fontsize=20) #yaxis-sigma*e^2/4hbar
ax1.set_ylabel('$\u03c3$',fontsize=20)

#ax1.set_ylim(-2.5e16,3.5e16)
#ax2.set_ylim(-30,20)
#ax2.set_ylabel('Im(${\u03B5_r}$)',fontsize=15)
ax1.legend(loc='best',frameon=False,fontsize=20) 
plt.show()


