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

def nose_hoover_chain(x0, v0, T, Q1, M=2, nc=1, nys=3, f0=None, m=1, omega=1, dt=0.01, nsteps=1000):
    '''
    Velocity Verlet integration for Langevin thermostat
    '''

    assert M >= 1
    assert nc >= 1
    assert nys == 3 or nys == 5

    if nys == 3:
        tmp = 1 / (2 - 2**(1./3))
        wdti = np.array([tmp, 1 - 2*tmp, tmp]) * dt / nc
    else:
        tmp = 1 / (4 - 4**(1./3))
        wdti = np.array([tmp, tmp, 1 - 4*tmp, tmp, tmp]) * dt / nc

    Qmass = np.ones(M) * Q1
    # if M > 1: Qmass[1:] /= 2
    Vlogs = np.zeros(M) 
    # Vlogs = np.sqrt(T / Qmass)
    Xlogs = np.zeros(M)
    Glogs = np.zeros(M)
    # for ii in range(1, M):
    #     Glogs[ii] = (Qmass[ii-1] * Vlogs[ii-1]**2 - T) / Qmass[ii]

    if f0 is None:
        f0 = harmonic_for(x0, m, omega)

    log(x0, v0, m, omega)


    def nhc_step(v, Glogs, Vlogs, Xlogs):
        '''
        '''

        scale = 1.0
        M     = Glogs.size
        K     = Ekin(v, m)
        K2    = 2*K
        Glogs[0] = (K2 - T) / Qmass[0]

        for inc in range(nc):
            for iys in range(nys):
                wdt = wdti[iys]
                # update the thermostat velocities
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt

                for kk in range(M-1):
                    AA = np.exp(-0.125 * wdt * Vlogs[M-1-kk])
                    Vlogs[M-2-kk] = Vlogs[M-2-kk] * AA * AA \
                                  + 0.25 * wdt * Glogs[M-2-kk] * AA

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * Vlogs[0])
                scale *= AA
                # update the forces
                Glogs[0] = (scale * scale * K2 - T) / Qmass[0]
                # update the thermostat positions
                Xlogs += 0.5 * Vlogs * wdt
                # update the thermostat velocities
                for kk in range(M-1):
                    AA = np.exp(-0.125 * wdt * Vlogs[kk+1])
                    Vlogs[kk] = Vlogs[kk] * AA * AA \
                              + 0.25 * wdt * Glogs[kk] * AA
                    Glogs[kk+1] = (Qmass[kk] * Vlogs[kk]**2 - T) / Qmass[kk+1]
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt

        return v * scale

    # 0, 1, 2 represents t, t + 0.5*dt, and t + dt, respectively
    for ii in range(nsteps):
        vnhc = nhc_step(v0, Glogs, Vlogs, Xlogs)
        v1   = vnhc + 0.5 * dt * f0 / m
        x2   = x0 + v1 * dt
        f2   = harmonic_for(x2, m, omega)
        v2p  = v1 + 0.5 * dt * f2 / m
        v2   = nhc_step(v2p, Glogs, Vlogs, Xlogs)

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
    T0 = 0.1
    v0 = np.sqrt(2*T0) * 2
    x0 = 0.0
    dt = 0.1
    N  = 20000

    np.random.seed(20210605)
    # vv(x0, v0, dt=0.1, nsteps=N)
    # vv(x0, v0, dt=0.01, nsteps=N)
    # vv(x0, v0, dt=0.005, nsteps=N)
    # nose_hoover(x0, v0, T=T0, Q=0.1, dt=0.1, nsteps=N)
    nose_hoover_chain(x0, v0, T=T0, Q1=0.1, M=1, dt=0.1, nsteps=N)
    nose_hoover_chain(x0, v0, T=T0, Q1=0.1, M=2, nc=1, dt=0.1, nsteps=N)
    # langevin(x0, v0, T=T0, gamma=0 / (np.pi * 2), dt=0.1, nsteps=N)
    # langevin(x0, v0, T=T0, gamma=1, dt=0.1, nsteps=20000)
    # langevin(x0, v0, T=T0, gamma=1000 / (np.pi * 2), dt=0.01, nsteps=N)
