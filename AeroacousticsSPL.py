#
# code Aeroacoustics PSD
"""
 @author : D. Chibouti
 Paris 2023
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sc
from scipy import integrate

# Constantes
# Fluid proprieties
rho = 1.2249                        # density de l'air
Temp= 288.15                        # temperature
mu  = 1.813*pow(10,-5)              # viscosity dynamique
nu  = mu/rho                        # viscosity cinematique

# Flight propieties
U_inf = 15.3 # 231.43               # velocity infty (en amont)
U_c  = 0.8 * U_inf                  # velocity de la couche limite at Trailing Edge (TE) 
c   = 0.15                          # corde (chord airfoil)
AoA = 9.32e-1                       # angle of attaque


Re = U_inf*c/nu                     # nombre de Reynolds
Ma = 0.0045                          # Mach number 
print('Reynolds of this case :', Re)


# XFOIL output parameters
## XFOIL output parameters for Suction Side (SS) 
theta_ss =  0.011968                   # Momentum thickness
Dstar_ss = 0.035869                     # Displacement thickness
shape_factor_ss = 2.997                # Shape factor
delta_ss = Dstar_ss * 8                 # Boundary Layer thickness (BL)

U_ratio_ss = 0.92030
Ue_ss = U_inf *(U_ratio_ss)            # Velocity at the boundary-layer edge
Cf_ss = 0.000195                       # Skin friction coefficient at TE
Tw_ss = 0.5*rho*U_inf**2 * Cf_ss       # Shear stress at TE due to BL

Rt_ss = 0.11* pow(Ue_ss*theta_ss/nu,0.75)

## XFOIL output parameters for Pressure Side (PS)
theta_ps = 0.004334                  # Momentum thickness
Dstar_ps = 0.007812             # Displacement thickness
shape_factor_ps = 1.802              # Shape factor
delta_ps = Dstar_ps * 8         # Boundary Layer thickness

U_ratio_ps = 0.92030             
Ue_ps = U_inf *(U_ratio_ps)          # Velocity at the boundary-layer edge
Cf_ps = 0.002566                     # Skin friction coefficient at TE
Tw_ps = 0.5*rho*U_inf**2 * Cf_ps     # Shear stress at TE due to BL

Rt_ps = 0.11* pow(Ue_ps*theta_ps/nu,0.75)


# Frequency range
freq_s = 100                        # start frequency omega_s
freq_e = (10000 + 1)                # end frequency omega_e
freq_range = np.arange(freq_s, freq_e)

# Calculating phi_pp for SS and PS
phi_pp_ss = np.zeros(freq_e - freq_s)         # SS
phi_pp_ps = np.zeros(freq_e - freq_s)       # PS
inc = 0                                        # increment counter



for omega in range (freq_s, freq_e):
    # equation Goody pour surface pressure spectrum on SS
    k_ss = 2*np.pi*omega*delta_ss/Ue_ss
    phi_pp_ss[inc] = (Tw_ss*Tw_ss*delta_ss/Ue_ss) * (3*k_ss*k_ss) / ( pow(pow(k_ss,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt_ss,-0.57)*k_ss, 7) )

    # equation Goody pour surface pressure spectrum on PS
    k_ps = 2*np.pi*omega*delta_ps/Ue_ss
    phi_pp_ps[inc] = (Tw_ps*Tw_ps*delta_ps/Ue_ss) * (3*k_ps*k_ps) / ( pow(pow(k_ps,0.75) + 0.5, 3.7) + pow(1.1*pow(Rt_ps,-0.57)*k_ps, 7) )
    
    inc+=1                                    # counter increment


# Plotting phi_pp
x_axis_freq = np.linspace(freq_s, freq_e, freq_e - freq_s)
plt.figure(1)
plt.plot(np.log10((x_axis_freq*delta_ss)/Ue_ss), 10*np.log10(phi_pp_ss/(Tw_ss*Tw_ss*delta_ss/(Ue_ss))), label = 'SS')
plt.plot(np.log10((x_axis_freq*delta_ps)/Ue_ps), 10*np.log10(phi_pp_ps/(Tw_ps*Tw_ps*delta_ps/Ue_ps)), label = 'PS')
plt.xlabel("$\omega\delta/U_e$"); plt.ylabel("$10log(\Phi_{pp}*U_e/Tw^2\delta)$")
plt.legend()
plt.savefig('Aeroacoustics_phi_pp_plots.png') 
plt.close()



# Diffraction theory for TE noise

alpha = 1.6                     # correlation length constant
L = 0.08                        # span length of airfoil [m]
c_0 = 334                       # speed of sound [m/s]
M = U_c /c_0   #U_inf/c_0       # convection Mach number

inc = 0
S_pp_SS = np.zeros(freq_e - freq_s)
S_pp_PS = np.zeros(freq_e - freq_s)
S_pp_Total = np.zeros(freq_e - freq_s)


for omega in range (freq_s, freq_e):
        
    # Radiation integral function calculation
    K_x = (2*np.pi*omega*c)/(2*U_c)   #  omega/U_c or (omega*c)/(2*U_c)
    al = Ue_ss/U_c
    beta = np.sqrt(1-M**2)
    mu_bar = (2*np.pi*omega*c)/(2*c_0*beta**2)

    xr = 1
    yr = 0
    zr = 1
    sigma = np.sqrt(xr**2 + (beta**2)*(yr**2 + zr**2))

    k_bar = 2*np.pi*omega*c/(2*c_0)
    k_x = 2*np.pi*omega/U_c
    k_y = 2*np.pi*omega*yr/(c_0*sigma)
    k_y_bar = k_y*c/2
    k_x_bar = k_x*c/2
    kappa = np.sqrt(mu_bar**2 - (k_y_bar**2/beta**2))
    eps = 1/np.sqrt(1+(1/(4*kappa)))

    i = 0+1j                            #imaginary
    B = K_x - M * mu_bar + kappa
    C = K_x - mu_bar*(xr/sigma - M)
    O = kappa - mu_bar*xr/sigma
    
    Theta = np.sqrt((k_x_bar + mu_bar * M + kappa)/(al * k_x_bar + mu_bar * M + kappa))
    H = (1+i)*np.exp(-4*i*kappa)*(1-Theta**2)/(2*np.sqrt(np.pi)*(al-1)*k_x_bar*np.sqrt(B))
    
    Es1, Ec1 = sc.fresnel((2*B-2*C)*np.sqrt(2/np.pi))
    E1 = Ec1 + i*Es1

    Es2, Ec2 = sc.fresnel((2*B)*np.sqrt(2/np.pi))
    E2 = Ec2 + i*Es2

    Es3, Ec3 = sc.fresnel((4*kappa)*np.sqrt(2/np.pi))
    E3 = Ec3 + i*Es3
    E3_con = np.transpose(np.conjugate(E3))

    Es4, Ec4 = sc.fresnel((2*O)*np.sqrt(2/np.pi))
    E4 = Ec4 + i*Es4
    
    G  = (1+eps)*np.exp(i*(2*kappa+O))*np.sin((O-2*kappa))/(O-2*kappa) \
         + (1-eps)*np.exp(i*(O-2*kappa))*np.sin((O+2*kappa))/(O+2*kappa) \
         + (1+eps)*(1-i)*np.exp(4*i*kappa)*E3/(2*(O-2*kappa)) \
         - (1-eps)*(1+i)*np.exp(-4*i*kappa)*E3_con/(2*(O+2*kappa)) \
         + 0.5*np.exp(2*i*O)*np.sqrt(2*kappa/O)*E4*(((1-eps)*(1+i)/(O+2*kappa))-((1+eps)*(1-i)/(O-2*kappa)))


    I_1 = i*np.exp(2*i*C)*((1+i)*np.exp(-2*i*C)*np.sqrt(B/(B-C))*E1 - (1+i)*E2 + 1)/C

    cI_ps = (1-(1+i)*E3)*np.exp(4*i*kappa);
    cI_ps = np.real(cI_ps) + i*np.imag(cI_ps)*eps;

    I_ps = H*cI_ps + H*(-np.exp(-2*i*O)+i*(O + k_x_bar + M*mu_bar - k_bar)*G)

    I = abs(I_1 + I_ps)
    
    # Amiet's TE theory: Power spectral density
    S_pp_SS[inc] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / 2*np.pi*omega ) * phi_pp_ss[inc] * I**2
    S_pp_PS[inc] = ((k_bar * zr)/(2*np.pi*sigma**2))**2  * 2 * L * (alpha * U_c / 2*np.pi*omega ) * phi_pp_ps[inc] * I**2
    S_pp_Total[inc] = S_pp_SS[inc] + S_pp_PS[inc]
    
    inc+=1


# Calculating SPL

Pref = 2E-5                                         # reference pressure

SPL = 10*np.log10(S_pp_Total)-20*np.log10(Pref)     # dB
SPL_SS = 10*np.log10(S_pp_SS)-20*np.log10(Pref)     # dB
SPL_PS = 10*np.log10(S_pp_PS)-20*np.log10(Pref)     # dB

Sf = pow(SPL/10,10)

E = (freq_e-freq_s)*sum(Sf)/len(Sf)     # Energy in the frequency range
E_t = integrate.trapz(Sf,x_axis_freq)   # Total energy under the spectrum curve using trapezoidal integration
OASPL = 10*np.log10(E_t)                # Overall Sound Pressure Level, calculated as 10 * log10 of the total energy


print('E_t   : ', E_t)
print('OASPL : ', OASPL)



#Plotting PSD

plt.plot(np.log10(x_axis_freq), SPL_SS, '-', label = 'SS')
plt.plot(np.log10(x_axis_freq), SPL_PS, '-', label = 'PS')
plt.plot(np.log10(x_axis_freq), SPL,'-', label = 'SS + PS')
plt.legend()
plt.savefig('Aeroacoustics_SPL_plots.png') 
plt.close()

plt.plot((x_axis_freq), SPL_SS, '-', label = 'SS')
plt.plot((x_axis_freq), SPL_PS, '-', label = 'PS')
plt.plot((x_axis_freq), SPL,'-', label = 'SS + PS')
plt.legend()
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Sound Pressure Level (dB)')
plt.savefig('Aeroacoustics_PSD_plots.png') 
#plt.show()
plt.close()

