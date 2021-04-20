# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import pickle

solu = pickle.load(open("../data/solution/solu_modified_40*200_0900", 'rb'))

phi = solu["phi"]

import matplotlib.pyplot as plt

import numpy as np
r = np.linspace(0,3000, num = 200)

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
