#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:07:40 2022

@author: hirschbe
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import modules as mod

class SedCas():
    
    def __init__(self):
        
        self.cwd = os.getcwd()
        self.ls = os.listdir(self.cwd)
        
        # get climate and parameter files
        for file in self.ls:
            if file.endswith('.met'):
                self.climatefile = file
            elif file.endswith('.par'):
                self.paramfile = file
        
    def load_climate(self):
        
        C = pd.read_csv(self.climatefile, sep='\t')
        C.D = pd.to_datetime(C.D)
        C.index = C.D

        self.Pr = C.Pr
        self.Ta = C.Ta
        self.Rsw = C.Rsw
        
    def load_params(self):
        
        with open(self.paramfile, 'r') as file:
            f = file.readlines()
            for l in range(2,35):
                linestr = f[l]
                linels = linestr.split(sep=':')
                try:
                    exec('self.%s = %s'%(linels[0], linels[1]))
                except NameError:
                    pass

        # normalizing hillslope sediment storage by catchment area considering packing density
        self.shcap = self.shcap*(self.rho_dry/self.rho_b) / self.area * 10**-3
        
        # smallest possible sediment amount in debirs flow
        # NOTE: this is only a constrain for the model, the smallest modelled debris flow volume is given by qdf and smax_nodf
        self.mindf = self.minDF * self.smax_nodf / self.area * 10**-3
        
    def run_hydro(self):
        
        # running the individual HRUs
        snow = list()
        PET = list()
        hydro = list()
        for i in range(self.n_HRU):
            s = mod.degree_day_model(self.Ta.copy(), self.Pr.copy(), self.mrate, self.Tsa, self.Tsm, s0=0, Asnow = self.Asnow[i], Asoil = self.Anosnow[i])
            snow.append(s)
            pet = mod.ET_PT(1, self.Rsw, self.Ta, 1, s.albedo, self.Ele, 0.8, 1, 1, 1, 0, 0)
            PET.append(pet)
            h = mod.hydmod(s, pet, self.Pr, self.Ta, self.alphaET, len(self.Vwcaps[i]), {'k':self.ks[i], 'Scap':self.Vwcaps[i], 'S0':[0,0]})
            hydro.append(h)

        # lumped hydrology: adding individual HRUs
        hyd = pd.DataFrame(columns = hydro[0].columns, index =  hydro[0].index)
        for c in hyd:
            try:
                lumped = list()
                for i in range(self.n_HRU):
                    l = hydro[i][c].values * self.shares[i]
                    lumped.append(l)
                lumped2 = np.array(lumped)
                hyd[c] = np.sum(lumped2, axis=0)
            except KeyError:
                pass
            if 'Vw' in c:
                if not c == 'Vw':
                    hyd.drop(columns = [c], inplace=True)
                    
        self.hydro = hyd
        
    def run_sediment(self):
        
        # initialization of variables for stochastic sediment supply

        n = len(self.Pr)                                                   # length of time series
        n_days = len(self.Pr.resample('24h').sum())                        # number of days 

        class sed:
            pass
        sed.index = self.Pr.index                                      # index for date
        sed.ls = np.zeros([n, self.M])                                 # sediment input from landslides
        sed.sc = np.zeros([n, self.M])                                 # sediment channel storage [mm], shape(time series length, number of simulations)
        sed.sh = np.zeros([n, self.M])                                 # sediment hillslope storage [mm]
        sed.so = np.zeros([n, self.M])                                 # sediment catchment output [mm]
        sed.sopot = np.zeros([n, self.M])                              # potential sediment catchment output [mm], i.e. transport-limited case
        sed.dfs = np.zeros([n, self.M])                                # debris flows, from 'so' values above threshold and summed consecutive values
        sed.dfspot = np.zeros(n)                                       # debris flows potential (only 1 because only 1 climate)
        
        # sediment module with stochastic landslide magnitudes
        print('running sediment module...', sep='\n', end='\n')
        seed = 0
        for m in tqdm(range(self.M)):  
            lrgls = mod.large_ls(self.Ta, self.Pr, self.hydro.snow, self.Tsd, self.Tpr, self.Tsa, self.ls_xmin, self.ls_alpha, self.ls_cutoff, self.Tfreeze, self.LStrig, self.area, seed=seed)                # large landslided
            N = len(lrgls[lrgls.mag > 0])
            sls = mod.small_ls(n_days, N, self.ls_xmin, self.area, seed=seed)                         # small landslides
            sls.index = lrgls.index                                     # date index for small landslides
            sed_run = mod.sedcas(lrgls, sls, self.hydro, self.qdf, self.smax, self.rhc, self.shcap, self.area, 'exp', self.LStrig, self.Tpr, shinit=self.shcap, mindf=self.mindf, smax_nodf=self.smax_nodf, b=self.b)
            sed.ls[:,m] = sed_run.ls.values
            sed.sc[:,m] = sed_run.sc
            sed.sh[:,m] = sed_run.sh
            sed.so[:,m] = sed_run.so
            sed.sopot[:,m] = sed_run.sopot
            sed.dfs[:,m] = sed_run.dfs
            seed += 1
        sed.dfspot[:] = sed_run.dfspot
            
        self.sed = sed
            
    def save_output(self):
        
        self.hydro.to_csv('Hydro.out', header=True)
        
        sedout = pd.DataFrame(index=self.sed.index)
        quants = [0, 10, 25, 50, 75, 90, 100]
        for q in quants:
            c = 'Q%i'%q
            sedout[c] = np.percentile(self.sed.so, q, axis=1)
        sedout.to_csv('Sediment.out', header=True)
        
    def plot_sedyield_monthly(self, save=True):
        
        cf = (self.area*10**6) * 10**-3 # km2 to m2 and mm to m
        
        sy = pd.DataFrame(data = self.sed.dfs*cf, index=self.sed.index)
        syp = pd.DataFrame(data = self.sed.dfspot*cf, index=self.sed.index)
        
        # monthly sediment yields
        sym = sy.resample('m').sum()
        sypm = syp.resample('m').sum()
        
        # mean monthly sediment yield
        symm = sym.groupby(by=sym.index.month).mean()
        sypmm = sypm.groupby(by=sypm.index.month).mean()

        # quantiles
        Q = [0,10,25,50,75,90,100] # quantiles
        SY = pd.DataFrame(index=symm.index)
        SYP = pd.DataFrame(index=sypmm.index, data=sypmm.values[:,0]) # only one potential since only one climate
        
        for q in Q:
            SY['Q'+str(q)] = np.percentile(symm.values, q,axis=1)
            # SYP['Q'+str(q)] = np.percentile(sypmm.values, q,axis=1)
        
        # # plot
        ca = 'steelblue' # color actual sedyield
        cp = 'darkred' # color potential sedyield
        x = np.arange(1,13)
        
        fig, ax = plt.subplots()
        ax.fill_between(x, SY.Q25, SY.Q75, color=ca, alpha=.5)
        ax.plot(x, SY.Q10, color=ca, ls='--', alpha=.5)
        ax.plot(x, SY.Q90, color=ca, ls='--', alpha=.5)
        ax.plot(x, SY.Q50, color=ca, lw=2)
        
        ax.plot(x, SYP.values, color=cp, lw=1, zorder=-1)
        
        # # legend
        l1 = plt.Line2D(x, SY.Q50, color=ca, lw=2, label = 'median supply-limited')
        l2 = plt.Line2D(x, SY.Q90, color=ca, ls='--', alpha=.5, label = 'Q10/Q90 supply-limited')
        l3 = mpatches.Patch(color=ca, label='Q25-Q75 supply-limited')
        l4 = plt.Line2D(x, SYP, color=cp, lw=2, label='transport-limited')
        ax.legend(handles=[l1, l2, l3, l4], fontsize='small', frameon=False)
        
        # # ax limits
        ax.set_xlim(1,12)    
        ax.set_ylim(0,)

        # # axis labels
        ax.set_xlabel('Months', fontsize=12)
        ax.set_ylabel('Debris-flow yield [m$^3$/month]', fontsize=12)    
        
        # # scientific notation
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
        ax.get_yaxis().get_offset_text().set_visible(False)
        exponent_axis = 3
        ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(0.0, 1.005), xycoords='axes fraction', fontsize=6) 
        
        if save:
            fig.savefig('MonthlySedYield.png')
