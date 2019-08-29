#!/usr/bin/env python
"""
Defines a class to integrate the density of the mean-field limit of the dWASEP
with arbitrary (variable) rate profile and  (fixed) Dirichlet boundary conditions.

Also works as stand-alone program:
> python dWASEP.py
"""

import numpy as np

__author__ = "Massimo Cavallaro"

__license__ = "GPLv3"

__version__ = "0.9"

__email__ = ["m.cavallaro@warwick.ac.uk", "cavallaromass@gmail.com"]


class DWASEP:

    def __init__(self, rho0, rhoL, rhoR, T=1000):

      # physical and numerical parameters:
      self.N = len(rho0)
      self.deltat = 0.002
      self.deltax = 0.05
      self.T = T

      # model parameters:
      # initial conditions
      self.rho0 = rho0
      # boundary conditions
      self.rhoL = rhoL  
      self.rhoR = rhoR

      self.robinL = - 2 - 2 * (2 * self.rhoL - 1) * self.deltax
      self.robinR = - 2 + 2 * (2 * self.rhoR - 1) * self.deltax

      self.u = np.zeros([self.T, self.N])
      self.h = np.zeros([self.T, self.N])
      self.rho = np.zeros([self.T, self.N])

      # # calculate current only in the bulk for convenience (optional)
      # self.J = np.zeros([self.T, self.N - 2])

      # Cole-Hopf transform initial conditions
      self.u[0,:] = np.exp(np.cumsum(2 * self.rho0 - 1) * self.deltax)

      # define bulk and ghost grid points
      self.x = np.arange(1, (self.N - 1))
      self.xm1 = np.arange(0, (self.N - 2))
      self.xp1 = np.arange(2, self.N)
    
    def integrate_pde(self, rate_profile):

      if len(rate_profile) != len(self.rho0):
        raise ValueError("the rate profile must have the same lenght as the the initial density profile")

      D = 0.5 * rate_profile

      r = max(D) * self.deltat / self.deltax ** 2

      if r > 0.5:
        print("The integration scheme is unstable, with r=%f"%r)


      for t in range(1, self.T):
          self.u[t, self.x] = self.u[t - 1, self.x] + self.deltat * \
                                    D[self.x] * (( self.u[t - 1, self.xm1] + self.u[t - 1, self.xp1] - 2 * self.u[t - 1, self.x]) / self.deltax ** 2 - self.u[t - 1, self.x])

          # Robin boundary conditions:
          self.u[t, 0] = self.u[t - 1, 0] + self.deltat * \
                                    D[1] * ((self.robinL * self.u[t - 1, 0] + 2 * self.u[t - 1, 1]) / self.deltax ** 2 - self.u[t - 1, 1])

          self.u[t, self.N - 1] = self.u[t - 1, self.N - 1] + self.deltat * \
                                    D[-1] * ((2 * self.u[t - 1, self.N - 2] + self.robinR * self.u[t - 1, self.N - 1]) / self.deltax ** 2 - self.u[t - 1, self.N - 1])

          # # compute current profile (optional)
          # self.J[t, :] = D[self.x] .* (0.5 - 0.5 * (self.u([t, self.xm1] + self.u[t, self.xp1] - 2 * self.u[t, self.x]) / (2 * self.deltax) / self.u[t, self.x]))


      # inverse Cole-Hopf transform
      self.h[:] = np.log(self.u) * 0.5
      # bulk:
      self.rho[:, self.x] = (self.h[:, self.xp1] - self.h[:, self.xm1]) / (2 * self.deltax) + 0.5
      # boundaries:
      self.rho[:, 0] = self.rhoL
      self.rho[:, self.N - 1] = self.rhoR



    def plot_rho(self):
      "create axis, plot rho, return the axis"
      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      X = np.arange(0, self.N)
      T = np.arange(0, self.T)
      X, T = np.meshgrid(X, T)
      ax.plot_surface(X, T, self.rho, linewidth=0, antialiased=False)
      ax.set_xlabel(r'$x$')
      ax.set_ylabel(r'$t$')
      ax.set_zlabel(r'$\rho$')
      return fig, ax


    def plot_u(self):
      "create axis, plot the height, return the axis"
      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      X = np.arange(0, self.N)
      T = np.arange(0, self.T)
      X, T = np.meshgrid(X, T)
      ax.plot_surface(T, X, self.u, linewidth=0, antialiased=False)
      ax.set_xlabel(r'$x$')
      ax.set_ylabel(r'$t$')
      ax.set_zlabel(r'$u$')
      return fig, ax

    def plot_J(self):
      "create axis, plot the current, return the axis"
      pass


if __name__ == '__main__':

    rho0 = np.hstack([np.repeat(1,50), np.repeat(0,100)])
    rhoL = rho0[0]
    rhoR = rho0[-1]

    dWASEP = DWASEP(rho0, rhoL, rhoR)

    rate_profile = np.hstack([
                      np.repeat(0.9048, 25),
                      np.exp(0.1 * np.sin(np.linspace(- np.pi / 2, - np.pi / 2 + 6 * np.pi, 100))),
                      np.repeat(0.9048, 25)]) - 0.1

    dWASEP.integrate_pde(rate_profile)

    # after calling the integrator,
    # to get the density profile just use:
    # dWASEP.rho

    print("the final density profile is:")
    print(dWASEP.rho[-1,:])

    fig, ax = dWASEP.plot_rho()
    fig.show()
    fig.savefig('rho.png')
