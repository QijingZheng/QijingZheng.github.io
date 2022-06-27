#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def harmonic_pot(x, m=1, omega=1):
    '''
    The harmonic potential
    '''

    return 0.5 * m * omega**2 * x**2

def harmonic_for(x, m=1, omega=1):
    '''
    The force of harmonic potential
    '''

    return - m * omega**2 * x

def Ekin(v, m=1):
    '''
    The force of harmonic potential
    '''

    return 0.5 * m * v**2

def log(x, v, m=1, omega=1):
    '''
    '''

    print("{:20.8E} {:20.8E} {:20.8E} {:20.8E} {:20.8E}".format(
        x, v,
        Ekin(v, m), 
        harmonic_pot(x, m, omega),
        harmonic_pot(x, m, omega) + Ekin(v, m)
    ))


def vv(x0, v0, f0=None, m=1, omega=1, dt=0.01, nsteps=1000):
    '''
    Velocity Verlet integration
    '''

    if f0 is None:
        f0 = harmonic_for(x0, m, omega)

    log(x0, v0, m, omega)

    for ii in range(nsteps):
        # x(t+dt) = x(t) + v(t) * dt + f(t) * dt**2 / (2*m)
        x1 = x0 + v0 * dt + f0 * dt**2 / (2*m)
        # f(t+dt)
        f1 = harmonic_for(x1, m, omega)
        # v(t+dt) = v(t) + dt * (f(t) + f(t+dt)) / (2*m)
        v1 = v0 + dt * (f1 + f0) / (2*m)

        log(x1, v1, m, omega)
        x0, v0, f0 = x1, v1, f1
        
def nose_hoover(x0, v0, T, Q, f0=None, m=1, omega=1, dt=0.01, nsteps=1000):
    '''
    Velocity Verlet integration for Langevin thermostat
    '''

    if f0 is None:
        f0 = harmonic_for(x0, m, omega)

    log(x0, v0, m, omega)

    eta0 = 0

    # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
    for ii in range(nsteps):
        x2   = x0 + v0 * dt + 0.5 * dt**2 * (f0 / m - eta0 * v0)
        v1   = v0 + 0.5 * dt * (f0 / m - eta0 * v0)
        f2   = harmonic_for(x2, m, omega)
        eta1 = eta0 + (dt / 2 / Q) * (0.5 * m * v0**2 - 0.5 * T)
        eta2 = eta1 + (dt / 2 / Q) * (0.5 * m * v1**2 - 0.5 * T)
        v2   = (v1 + 0.5 * dt * f2 / m) / (1 + 0.5 * dt * eta2)

        log(x2, v2, m, omega)
        x0, v0, f0 = x2, v2, f2

def langevin(x0, v0, T, gamma, dt=0.01, f0=None, m=1, omega=1, nsteps=1000):
    '''
    Velocity Verlet integration for Nose-Hoover thermostat
    '''

    if f0 is None:
        f0 = harmonic_for(x0, m, omega)

    log(x0, v0, m, omega)

    # the random force is sampled from Gaussian distribution with variance
    # sigma**2
    sigma = np.sqrt(2 * m * gamma * T / dt)
    R0    = np.random.standard_normal() * sigma

    # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
    for ii in range(nsteps):
        v1 = v0 + 0.5 * dt * (f0 - gamma * m * v0 + R0) / m
        x2 = x0 + v1 * dt
        f2 = harmonic_for(x2, m, omega)
        R2 = np.random.standard_normal() * sigma
        v2 = v1 + 0.5 * dt * (f2 - gamma * m * v1 + R2) / m

        log(x2, v2, m, omega)
        x0, v0, f0, R0 = x2, v2, f2, R2

if __name__ == "__main__":
    T0 = 1
    v0 = np.sqrt(2*T0)
    x0 = 0.0
    dt = 0.05
    N  = 100000

    np.random.seed(20210605)
    # vv(x0, v0, dt=dt, nsteps=N)
    # nose_hoover(x0, v0, T=T0, Q=0.2, dt=dt, nsteps=N)
    langevin(x0, v0, T=T0, gamma=0 / (np.pi * 2), dt=dt, nsteps=N)
    langevin(x0, v0, T=T0, gamma=1 / (np.pi * 2), dt=dt, nsteps=N)
    langevin(x0, v0, T=T0, gamma=1000 / (np.pi * 2), dt=dt, nsteps=N)
