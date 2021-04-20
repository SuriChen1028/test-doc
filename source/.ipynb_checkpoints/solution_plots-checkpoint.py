# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: ry38
#     language: python
#     name: ry38
# ---

# # Plots for BHmodified.

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16

solu10 = pickle.load(open("../data/solution/solu_modified_20*200*200_10_0202", "rb"))
solu20 = pickle.load(open("../data/solution/solu_modified_20*200*200_20_0202", "rb")) 
solu40 = pickle.load(open("../data/solution/solu_modified_20*200*200_40_0202", "rb")) 

solu10

log_ell = np.linspace(-13, -5, 200)
ell_step = 1e-7


def compute_ell_r_phi(solu, log_ell=np.linspace(-13, -5, 200), ell_step=1e-7):
    z = np.linspace(1e-5, 2, 20)
    x_r,  = log_ell.shape
    y_r, = z.shape
    r = np.zeros((x_r, y_r))
    phi = np.zeros((x_r, y_r))
    ell_new = np.zeros(x_r)
    for i, ell in enumerate(np.exp(log_ell)):
        psi = solu[ell]["psi"][:, -1]
        psi_next = solu[ell+ell_step]["psi"][:, -1]
        dpsi = (psi_next - psi)/ell_step
        psi_new = (psi + psi_next)/2
        ell_new[i] = ell + ell_step/2
        r[i] = - dpsi
        phi[i] = psi_new + ell_new[i]*(-dpsi)
    
    index = np.argsort(r, axis=0)
    phi_sorted = phi[index[:, 0]]
    r_sorted = r[index[:, 0]]
    ell_sorted = ell_new[index[:, 0]]
    return ell_sorted, r_sorted, phi_sorted


ell10, r10, phi10 = compute_ell_r_phi(solu10)
ell20, r20, phi20 = compute_ell_r_phi(solu20)
ell40, r40, phi40 = compute_ell_r_phi(solu40)

phi40.shape, ell40.shape

plt.plot(r10[:, 10], ell10)

index = np.argsort(r, axis=0)
phi_sorted = phi[index[:, 0]]
r_sorted = r[index[:, 0]]
ell_sorted = ell_new[index[:, 0]]

plt.plot(ell10, r10[:, 7])
plt.plot(ell10, r10[:, 10])
plt.plot(ell10, r10[:, 13])

from constants import *
tau = MEDIAN*GAMMA_BASE
sigma_z = .21
rho = .5
mu_2 = 1.
sigma_2 = np.sqrt(2*sigma_z**2*rho/mu_2)
sigma_2, XI_m

z[7]


# compute dphi_dr
def compute_dphidr(phi, r):
    dphi = (phi[1:] - phi[:-1])/(r[1:] - r[:-1])
    r_new = (r[1:] + r[:-1])/2
    phi_new = (phi[1:] + phi[:-1])/2
    ems = DELTA*ETA/(tau*z + dphi)
    return r_new, phi_new, ems


def compute_dphidz(phi, z = np.linspace(1e-5, 2, 20)):
    dphi_dz = (phi[:, 1:] - phi[:, :-1])/(z[1:] - z[:-1])
    z_new = (z[1:] + z[ :-1])/2
    return z_new, dphi_dz


r_new10, phi_new10, ems10 = compute_dphidr(phi10, r10)
r_new20, phi_new20, ems20 = compute_dphidr(phi20, r20)
r_new40, phi_new40, ems40 = compute_dphidr(phi40, r40)

plt.plot(ems10)
plt.plot(ems20)

z_new10, dphi_dz10 = compute_dphidz(phi10)
z_new20, dphi_dz20 = compute_dphidz(phi20)
z_new40, dphi_dz40 = compute_dphidz(phi40)

dphi_dz10.shape, z_new10.shape

plt.plot(r10[:, 0], dphi_dz10[:, 0])

XI_m

# +
h_z10 = - dphi_dz10*z_new10*sigma_2**2/(XI_m/10)

h_z20 = - dphi_dz20*z_new20*sigma_2**2/(XI_m/20)
h_z40 = - dphi_dz40*z_new40*sigma_2**2/(XI_m/40)
# h_z50 = - dphi_dz*z_new*sigma_2**2/(XI_m/50)
# -

h_z40.shape, ell40.shape

plt.plot(ell10, h_z10[:, 7])
plt.plot(ell10, h_z10[:, 10])
plt.plot(ell10, h_z10[:, 13])

fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(ell10, h_z10[:, 13], label=r'90 percentile of $z_2$')
plt.plot(ell10, h_z10[:, 10], label=r'50 percentile of $z_2$')
plt.plot(ell10, h_z10[:, 7], label=r"10 percentile of $z_2$")
plt.legend(title=r"with $\xi_m=.000256$")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad = 45, rotation = 0)
plt.title("implied distortion - multiplier")
# plt.savefig("../figures/h_ell_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()

fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(ell40, h_z40[:, 10], label=r'$\xi_m/40$')
plt.plot(ell20, h_z20[:, 10], label=r"$\xi_m/20$")
plt.plot(ell10, h_z10[:, 10], label=r"$\xi_m /10$")
plt.legend(title=r"original $\xi_m$ = .00256, median $z_2$")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad = 45, rotation = 0)
plt.title("implied distortion - multiplier")
# plt.savefig("../figures/h_ell_xi_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()

fig = plt.figure(figsize=(12, 8), dpi=100)
plt.plot(r40[:, 10], h_z40[:, 10], label=r'$\xi_m/10$')
plt.plot(r20[:, 10], h_z20[:, 10], label=r'$\xi_m/20$')
plt.plot(r10[:, 10], h_z10[:, 10], label=r'$\xi_m/40$')
plt.legend(title=r"original $\xi_m$ = .00256, median $z_2$ ")
plt.xlabel(r"$r$")
plt.ylabel(r"$\sqrt{z_2}\sigma_2 h_2$", labelpad = 45, rotation = 0)
plt.title("implied distortion - reserve")
# plt.savefig("../figures/h_r_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()

h_z40[:, 10][-1]

dphi_dz.shape, z_new.shape

plt.plot(r_new[:, 7][150:], phi_new[:, 7][150:], label='1')
plt.plot(r_new[:, 10][150:], phi_new[:, 10][150:], label=2)
plt.plot(r_new[:, 13][150:], phi_new[:, 13][150:], label=3)
plt.legend()
plt.show()


def simulate_emission(ems, r, r_initial=1500, time = 500):
    ems_t = np.zeros(time)
    r_remain = r_initial
    for i in range(time):
        loc = np.abs(r-r_remain).argmin()
        ems_i =  ems[loc]
        ems_t[i] = ems_i
        r_remain = r_remain - ems_i
    return ems_t


ems_10 = simulate_emission(ems10[:, 10], r10[:, 10])

plt.plot(ems_10)


def simulate_h(dphi_dz, z_grid, z_idx , r, ems, z = np.linspace(1e-5, 2, 20), r_initial = 1500, time = 500, xi = XI_m):
    r_remain = r_initial
    z_idx = int(z_idx)
    loc = np.abs(z_grid - z[z_idx]).argmin()    
    dphi = dphi_dz[:,loc]
    zvalue = z_grid[loc]
    print(zvalue)
    h_t = np.zeros(time)
    for i in range(time):
        loc = np.abs(r-r_remain).argmin()
        ems_i = ems[loc]
        dphi_dz_i = dphi[loc]
        h_t[i] = - dphi_dz_i*zvalue*sigma_2**2/xi
        r_remain = r_remain - ems_i
    return h_t


h_10 = simulate_h(dphi_dz10, z_new10, 10, r10[:,10], ems10[:,10], xi=XI_m/10)
h_7 = simulate_h(dphi_dz10, z_new10, 7, r10[:,7], ems10[:,7], xi =XI_m/10)
h_13 = simulate_h(dphi_dz10, z_new10, 13, r10[:,13], ems10[:,13], xi = XI_m/10)

z_new[7], z_new[10], z_new[13]

fig = plt.figure(figsize = (12,8), dpi = 100)
plt.plot(h_13, label = r"90th percentile of $z_2$")
plt.plot(h_10, label = r"50th percentile of $z_2$")
plt.plot(h_7, label = r"10th percentile of $z_2$")
plt.legend(title=r"original $\xi_m$ = .000256 ")
plt.xlabel("year")
plt.ylabel("implied distortion")
# plt.savefig("../figures/ht_0201.png", bbox_inches='tight', facecolor = "white")
plt.show()

h10 = simulate_h(dphi_dz10, z_new10, 10, r10[:,10], ems10[:,10], xi = XI_m/10)
h20 = simulate_h(dphi_dz20, z_new20, 10, r20[:,10], ems20[:,10], xi = XI_m/20)
h40 = simulate_h(dphi_dz40, z_new40, 10, r40[:,10], ems40[:,10], xi = XI_m/40)

z_new10

fig = plt.figure(figsize = (12,8), dpi = 100)
plt.plot(h40, label = r"$\xi_m/40$")
plt.plot(h20, label = r"$\xi_m/20$")
plt.plot(h10, label = r"$\xi_m/10$")
plt.legend(title = r"original $\xi_m$ = .00256, median $z_2$ ")
plt.xlabel("year")
plt.ylabel("implied distortion")
plt.title("implied distortion - time")
# plt.savefig("../figures/ht_0202.png", bbox_inches='tight', facecolor = "white")
plt.show()

z[10] - 3

ems_10 = simulate_emission(ems[:, 10], r_new[:, 10])
ems_7 = simulate_emission(ems[:, 7], r_new[:, 7])
ems_13 = simulate_emission(ems[:, 13], r_new[:, 13])

z[10],z[7], z[13]

sigma_z= .21
z[10]- sigma_z*3

fig = plt.figure(figsize = (12,8), dpi = 100)
plt.plot(ems_7, label = r"$z_2$ = .73")
plt.plot(ems_10, label = r"$z_2$ = 1.05")
plt.plot(ems_13, label = r"$z_2$ = 1.37")
plt.legend()
plt.xlabel("year")
plt.ylabel("emission")
plt.savefig("../figures/et_0201.png", bbox_inches='tight', facecolor = "white")
plt.show()

r_new.shape, phi_new.shape

ems.shape

plt.plot(r[:,0])
plt.plot(r_sorted[:,0])
plt.plot(r_new[:,0])

phi = solu["phi"]

dv = solu["dvdy"]
dv[0]

import matplotlib.pyplot as plt
import matplotlib as mpls
# mpl.rcParams['font.size']=16
fig = plt.figure(figsize = (12,8), dpi = 100)
plt.plot(y[:100], phi[-1][:100], label = r"$z = 10^{-5}$")
# plt.plot(y, dv[10], label = r"$z = 10^{-5}$")
plt.xlabel("y")
plt.ylabel(r"$\frac{d\phi}{dy}$", rotation=0, fontsize = 20)
plt.title("$z = 10^{-5}$, 1000th iteration, y $\in$ [0, 1500]")
# plt.savefig("../figures/dvdy_1500.png", bbox_inches='tight', facecolor = "white")
plt.show()

# mpl.rcParams['font.size']=16
fig = plt.figure(figsize = (12,8), dpi = 100)
plt.plot(y, phi[0], label = r"$z = 10^{-5}$")
plt.plot(y, phi[10], label = r"$z = 2$")
plt.hlines(0, xmin=-100, xmax = 3000, linestyle = "dashed", color = "black")
plt.xlabel("y")
plt.ylabel(r"$\hat{\phi}(y,z)$", rotation=0, fontsize = 20, x=-200)
plt.title("After the first derivative")
plt.legend()
C
plt.show()

import numpy as np
z = np.linspace(1e-5, 2, num = 20)
z[0], z[7], z[-1]
y = np.linspace(0, 3000, num = 200)

import requests
import matplotlib as mpl
mpl.rcParams['font.size']=14
# plt.plot(solu[0,:,0][:300], solu[0,:,1][:300])
plt.plot(solu[4,:,0][300:], solu[4,:,1][300:], label="z = 2.1")
plt.plot(solu[7,:,0][300:], solu[7,:,1][300:], label = "z = 3.68")
# plt.plot(solu[-1,:,0], solu[-1,:,1], label = "z = 10")
plt.legend()
plt.xlabel("r")
plt.ylabel(r"$\phi(r)$")
plt.savefig("../figures/phi_r_0128.png", bbox_inches='tight', facecolor = "white")

import numpy as np
r = np.linspace(0,3000, num = 200)

y = .032*np.log(np.exp(.00175*.018*r/.032) - .0001) -  .00175*.018*r

plt.plot(y)

z =  z = np.linspace(1e-5, 2, num=40) 
z[0], z[1], z[-1]

phi[0].shape

import matplotlib as mpl
mpl.rcParams['font.size']=20
fig = plt.figure(figsize = (12,8),dpi = 128)
plt.plot(r, phi[0], label = r"$z_2 = 10^{-5}$")
plt.plot(r, phi[1], label = r"$z_2  = 0.053 $")
plt.plot(r, phi[-1], label = r"$z_2 = 1$")
plt.legend()
plt.xlabel('reserve')
plt.ylabel(r'$\phi$')
# plt.savefig("../figures/phi_r_0127.png", bbox_inches='tight', facecolor = "white")
plt.show()

# +
e = solu["e"]
fig = plt.figure(figsize = (24,8),dpi = 128)
ax1=plt.subplot(121)
# ax1.plot(r, e[0], label = r"$z_2 = 10^{-5}$")
ax1.plot(r, e[1], label = r"$z_2  = 0.053 $")
ax1.plot(r, e[-1], label = r"$z_2 = 2$", color="green")
plt.legend()
plt.xlabel('reserve')
plt.ylabel('emission')

# ax2 =plt.subplot(122)
# ax2.plot(r, e[-1], label = r"$z_2 = 1$", color = "green")
# plt.legend()
# plt.xlabel('reserve')
# plt.ylabel('emission')

# plt.savefig("../figures/e_r_0127.png", bbox_inches='tight', facecolor = "white")
plt.show()
# -

r[100]

e = solu["e"]
fig = plt.figure(figsize = (12,8),dpi = 128)
plt.plot(z, e[:, 0], label = r"$r = 0$")
plt.plot(z, e[:,1], label = r"$r  = 150 $")
plt.plot(z, e[:,-1], label = r"$r = 1500$")
plt.legend(fontsize = 20)
plt.xlabel(r'$z_2$')
plt.ylabel(r'Emission')
# plt.savefig("../figures/e_r.png", bbox_inches='tight', facecolor = "white")
plt.show()

plt.plot(phi[10]*100)

e[-1]

solu10_test = pickle.load(open("../data/solution/solu_modified_20*200*200_10_0203", "rb"))
solu10_test
