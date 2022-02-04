#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:07:40 2022

@author: hirschbe
"""

import pandas as pd
import numpy as np
import math
import random

#%% Snow Degree-Day-Module
def degree_day_model(T, P, m, Ta, Tm, s0=0, Asnow=0.8, Asoil=0.3):
    '''
    Computing snow depth in SWE based on a degree-day-snow-melt-model
    -------------------------------------------------
    
    Inputs
    ------
    T : Temperature time series [degreeC]
    P : Precipitation time series [mm]
    m : melt rate factor [mm/degreeC/t]
    Ta : temperature threshold for snow accumulation [degreeC]
    Tm : temperature threshold for snow melt [degreeC]
    s0 : initial snow depth [mm]
    Asnow : snow albedo [-]
    Asoil : soil albedo [-]
    
    Output
    ------
    df : dataframe containing
        - ddepth : daily snow pack change SWE [mm]
        - depth : modelled SWE snow depth SWE [mm]  
        - acc : snow pack accumulation SWE [mm]
        - melt : melt from snow pack SWE [mm]
        
    Note: acc and melt is not necessary equal to ddepth because there can be both
    at the same time step (if accumulation and melt temperature threshold are not equal)
    '''
    index = P.index
    
    T = T.values
    P = P.values
    
    # snow accumulation
    cond = T <= Ta
    acc = P
    acc[~cond] = 0. # no snow pack accumulation where temperature is above
    
    # snow melt
    Tgrad = T - Tm # melting gradient
    cond = T > Tm
    Tgrad[~cond] = 0. # no snowmelt where temperature is below
    melt = m * Tgrad # potential snowmelt based on temperature
    
    # compute actual snow depth
    depth = np.zeros(len(P))
    ddepth = np.zeros(len(P))
    for i in range(1,len(P)):
        depth[i] = depth[i-1] + acc[i] - melt[i]
        if depth[i] < 0:
            melt[i] = melt[i] - abs(depth[i])
            depth[i] = 0
        ddepth[i] = depth[i] - depth[i-1]  
        
    # Albedo
    A = np.zeros(len(depth))
    cond = depth > 0
    A[cond] = Asnow
    A[~cond] = Asoil
    
    data = {'depth' : depth,
            'ddepth' : ddepth,
            'sacc' : acc,
            'smelt' : melt,
            'albedo' : A
            }
    
    df = pd.DataFrame(data = data, index = index)
    
    return df

#%% Potential Evapotranspiration Module
def ET_PM_PT(dt,Rsw,Ta,N,A,Ele,U,OPT,d,z=np.nan,Ws=np.nan,rs_min=np.nan,zom=np.nan):
    '''
    # Function for calculation of Evapotranspiration using either
    # Penmann-Monteith or Priestly Taylor.

    # MATLAB code by Simone Fatichi
    -------------------------------
    INPUTS
    ------
    dt = [h], temporal resolution, e.g. [24] for daily
    U =[0-1], ? e.g. 0.8
    Rsw [W/m^2], radiation
    Ta [degreeC], time series temperature
    N [-], cloud cover fraction
    A = [-], Albedo time series
    Ele = [m a.s.l.]
    z = [m], e.g. 1
    zom =[m], e.g. 0
    Ws = [m/s], e.g. 1
    rs_min [s/m], e.g. 0
    OPT = mehtod, 'PT' or 'PM'
    d = ?, e.g. 1, not used in method 'PT'
    
    OUTPUTS
    -------
    EPT : PET [mm]
    Epot :
    Tpot :
    EP_en :
    EP_aer :
    '''
    esat=611*np.exp(17.27*Ta/(237.3+Ta))  ## Vapor Pressure Saturation
    ea = U*esat  
    #### Net Radiation 
    k=0.4  # von Karman constant 
    si = 5.6704*(10**-8)  # Stefan-Boltzman Constant [W/m**2.K**4]
    ######
    K = 0.1 +0.9*(1-0.6*(N**2.5))  #### Emissiivty coefficient cloud  
    ei= 0.34 -0.14*np.sqrt(ea/1000)  ## Net emissivity humidity 
    ######
    D_Rlw=ei*K*si*((Ta-273.15)**4)  ## Net Longwave Radiatio W/m^2 
    Rn = Rsw*(1-A) - D_Rlw  ## Net Radiation  W/m^2
    #################################################
    ##### COMBINED ENERGETIC AND AERODYNAMIC METHOD 
    ### PARAMETERS 
    #p_p0   = exp(-Ele/8434.5)  ### correction for differences in pressure between basin and seal level
    #Pre = P0*p_p0 
    Pre0=101325  ##[Pa] 
    Pre = Pre0*np.exp((-9.81/287)*(Ele-0)/(Ta+273.15)) #[Pa] 
    #############################5 
    ro = (Pre/(287.04*(Ta+273.15)))*(1-(ea/Pre)*(1-0.622))  ##  dry air density [kg/m^3]
    row = 1000  # water density [kg/m^3]
    cp=1005 + ((Ta +23.15)**2)/3364  ## specific heat air  [J/kg K]
    L= 1000*(2501.3 - 2.361*(Ta))  ### Latent heat vaporization/condensaition [J/kg]
    ##############
    ######################################
    if OPT == 'PM': #### PENMANN-MONTEITH
        #B=(0.622*(k^2)*ro*Ws)/(Pre*row*(log((z-d)/zom))^2)  ## [m/s.Pa]
        B=(0.622*(k**2)*ro*Ws)/(Pre*row*(np.log((z-d)/zom)*np.log((z-d)/(0.1*zom))))  ## [m/s.Pa]
        #ra= ((log((z-d)/zom))^2)/(Ws*k^2)  ##  ## aerodynamic resistence [s/m] Neutal Condition
        ra= (np.log((z-d)/zom)*np.log((z-d)/(0.1*zom)))/(Ws*k**2)  ##  ## aerodynamic resistence [s/m] Neutal Condition
        EP_en=1000*3600*Rn/(L*row)  ## [mm/h] evaporazione da bilancio energetico
        EP_aer=1000*3600*B*(esat-ea)  # [mm/h] evaporazione da bilancio aerodinamico
        G=cp*Pre/(0.622*L)  ## Pa/c costante psicrometrica
        D=(4098*esat)/((237.3+Ta)**2)  ## Pa/C
#        EP=(D/(D+G))*EP_en+(G/(D+G))*EP_aer  ## [mm/h] Penmann Monteith combined method
        ###############
        Tpot=(1000*3600/row)*(1/L)*(D*Rn + ro*cp*(esat-ea)/ra)/(D +G*(1+rs_min/ra))  ## Evapotraspirazione potenziale [mm/h]
        Epot=(1000*3600/row)*(1/L)*(D*Rn + ro*cp*(esat-ea)/ra)/(D +G)  ## Evaporazione potenziale [mm/h]
        ##### obviously here Epot == EP 
        EPT = 1.26*(D/(D+G))*EP_en  ## [mm/h] Priestly-Taylor 
    
    elif OPT=='PT': ######### PRIESTLY-TAYLOR
        EP_en=1000*3600*Rn/(L*row)  ## [mm/h] evaporation 
        G=cp*Pre/(0.622*L)  ## Pa/c  psicrometric costant
        D=(4098*esat)/((237.3+Ta)**2)  ## Pa/C
        EPT= 1.26*(D/(D+G))*EP_en  ## [mm/h] Priestly-Taylor 
        EP_aer = np.nan 
        Epot=np.nan  
        Tpot=np.nan
    else:
        raise TypeError('Supplied ET OPT was not recognised!')
    if not np.isclose(dt, 1.):
        EPT = EPT*dt  ##[mm/dt] 
        EP_aer = EP_aer*dt  
        EP_en = EP_en*dt 
        Tpot=Tpot*dt  
        Epot=Epot*dt
        
    # EPT may be negative due to dew in the winter. However, we do not consider dew and are just
    # interested in positive values.
    #EPT[EPT<0] = 0

    return EPT, Epot, Tpot, EP_en, EP_aer

def ET_PT(dt,Rsw,Ta,N,A,Ele,U,d,z=np.nan,Ws=np.nan,rs_min=np.nan,zom=np.nan):
    '''
    # Function for calculation of Evapotranspiration using either Priestly Taylor.

    # modified from: MATLAB code by Simone Fatichi
    -------------------------------
    INPUTS
    ------
    dt = [h], temporal resolution, e.g. [24] for daily
    U =[0-1], ? e.g. 0.8
    Rsw [W/m^2], radiation
    Ta [degreeC], time series temperature
    N [-], cloud cover fraction
    A = [-], Albedo time series
    Ele = [m a.s.l.]
    z = [m], e.g. 1
    zom =[m], e.g. 0
    Ws = [m/s], e.g. 1
    rs_min [s/m], e.g. 0
    OPT = mehtod, 'PT' or 'PM'
    d = ?, e.g. 1, not used in method 'PT'
    
    OUTPUTS
    -------
    EPT : PET [mm]
    Epot :
    Tpot :
    EP_en :
    EP_aer :
    '''
    esat=611*np.exp(17.27*Ta/(237.3+Ta))  ## Vapor Pressure Saturation
    ea = U*esat  
    #### Net Radiation 
    si = 5.6704*(10**-8)  # Stefan-Boltzman Constant [W/m**2.K**4]
    ######
    K = 0.1 +0.9*(1-0.6*(N**2.5))  #### Emissiivty coefficient cloud  
    ei= 0.34 -0.14*np.sqrt(ea/1000)  ## Net emissivity humidity 
    ######
    D_Rlw=ei*K*si*((Ta-273.15)**4)  ## Net Longwave Radiatio W/m^2 
    Rn = Rsw*(1-A) - D_Rlw  ## Net Radiation  W/m^2
    #################################################
    ##### COMBINED ENERGETIC AND AERODYNAMIC METHOD 
    ### PARAMETERS 
    #p_p0   = exp(-Ele/8434.5)  ### correction for differences in pressure between basin and seal level
    #Pre = P0*p_p0 
    Pre0=101325  ##[Pa] 
    Pre = Pre0*np.exp((-9.81/287)*(Ele-0)/(Ta+273.15)) #[Pa] 
    #############################5 
    row = 1000  # water density [kg/m^3]
    cp=1005 + ((Ta +23.15)**2)/3364  ## specific heat air  [J/kg K]
    L= 1000*(2501.3 - 2.361*(Ta))  ### Latent heat vaporization/condensaition [J/kg]
    ##############
    ######################################
    
    EP_en=1000*3600*Rn/(L*row)  ## [mm/h] evaporation 
    G=cp*Pre/(0.622*L)  ## Pa/c  psicrometric costant
    D=(4098*esat)/((237.3+Ta)**2)  ## Pa/C
    EPT= 1.26*(D/(D+G))*EP_en  ## [mm/h] Priestly-Taylor 
    EP_aer = np.nan 
    Epot=np.nan  
    Tpot=np.nan

    if not np.isclose(dt, 1.):
        EPT = EPT*dt  ##[mm/dt] 
        EP_aer = EP_aer*dt  
        EP_en = EP_en*dt 
        Tpot=Tpot*dt  
        Epot=Epot*dt
        
    # EPT may be negative due to dew in the winter. However, we do not consider dew and are just
    # interested in positive values.
    EPT[EPT<0] = 0

    return EPT


def ET_Hamon(J, T, phi):
    '''
    ALTERNATIVE CALCULATION OF EPT if you don't have data on solar radiation or cloud cover.
    Hamon PET from Hornberger and Wiberg: Numerical Methods in the Hydrological Sciences
    ----------------------------------------------------------------------------------------
    
    Inputs
    ------
    phi : latitude [degrees]
    J : array of Julian day
    T : array of Temperatures
    
    '''
    #phi = 46.27                                 # Latitude in degrees
    delta = 0.4093*math.sin((2*np.pi/365)*J-1.405)   # Solar declination
    omega_s = math.acos(-math.tan(2*np.pi*phi/360.)*math.tan(delta))   # sunset hour angle
    Nt = 24*omega_s/np.pi                       # hours in day
    a = 0.6108
    b = 17.27 
    c = 237.3
    es = a*np.exp(b*T/(T+c))                      # saturation vapor pressure
    PET = ((2.1*(Nt**2)*es)/(T+273.3))       # Hamont PT in mm/m2
    i_cold = T <= 0
    PET[i_cold] = 0
    
    return PET

#%% HYDROLOGICAL MODEL

def hydmod(snow, PET, Pr, Ta, a, l, params):
    '''
    ...work in progress...
    linear reservoir hydrological model accounting for snow, ET, soil water storage, discharge.
    Unlike the other hydmod module this one is extended by another bucket in the sence of an upper and a lower bucket. 
    -------------------------------------------------
    
    Inputs
    ------
    snow : data frame from degree-day-model
    PET : potential ET from PET module
    P : Precipitation
    T : Temperature
    a : parameter for efficiency of ET dependent on saturation of upper storage (not like before from Tuttle and Salvucci, 2012)
    l : number of reservoirs
    params : dict containing
        k : factor for release from linera reservoir, i.e. residence time
        Scap : water storage capacity [mm]
        S0 : initial condition [mm]
        example: if n=2 the input shoud be params = dict('k' : [k1, k2], 'Scap' : [sc1, sc2], 'S0' : [s01, s02])
        
    Output
    ------
    hyd : dataframe containing time series of...
        - Q : dischagre [mm]
        - Qs : discharge from overland flow [mm]
        - Qss : discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
        - Vw : state of soil water storage [mm]
        - snow : snow depth SWE [mm], already in input
        - PET : Potential ET [mm]
        - AET : Actual ET [mm]
    '''
    #%% initialization
    index = snow.index
    n = len(snow)
    sdepth = snow.depth.values
    dsdepth = snow.ddepth.values
    sacc = snow.sacc.values
    smelt = snow.smelt.values
    PET = PET.values
    Pr = Pr.values
    Ta = Ta.values
    
    Q, Qss, Qs = np.zeros(n), np.zeros(n), np.zeros(n) # dischage
    Qper = np.zeros(shape=(n,l-1)) # percolation between storages the first column is for the flow between the most upper and the second...
    
    # assign storage properties
    class storage(object):
        '''
        makes class from parameter inputs
        '''
        def __init__(self, dictionary):
            for key in dictionary:
                setattr(self, key, dictionary[key])
                
    s = storage(params)
    
    Vw = np.zeros((n, l))                              # array for storage time series, each column represents one reservoir (from bottom to top)
    AET = np.zeros(n)
    for i in range(l):
        Vw[0,i] = s.S0[i]                              # initial condition
    
    #%% transient storage, discharge computation
    # changes from precipitation and snowmelt (both positive, because they are inputs to the storage)
    # to keep this order is important because there can be snow accumulation and melt on the same day
    dVw = Pr.copy()                            # add precipitation
    dVw = dVw - sacc                          # when snow accumulation, precipitation is not added to the storage
    dVw = dVw + smelt                         # where snow melt, add it
    
    # loop through each time step
    ## scheme: 
        # 1) add everything to the most upper storage
        # 2) compute all internal and outflows based on state afer input
        # 3) subtract outflows, account for internal flow
        # 4) if lower is full, it is pushed to the upper
        # 5) check that capacity of upper is not exceeded, else more surface runoff
        
    # loop through time steps
    # i --> time index
    for i in range(1,n):
        # 1)
        Vw[i,0] = Vw[i-1,0] + dVw[i]
        
        # 2)
        # AET
        if Ta[i] > 0:
            b = 1. - np.exp(-a*Vw[i,0]/float(s.Scap[0]))
            AET[i] = b * PET[i]
        
        # this is out of temperature loop, should not make a difference, but ensures continuity of water just in case
        if Vw[i,0] > s.Scap[0]:
            Qs[i] = Vw[i,0] - s.Scap[0]         # the amount exceeding the capacity is surface runoff
        
        if Ta[i] > 0:                            # condition for subsurface flow, else frozen
            # percolation
            # j --> storage index
            for j in range(l-1):
                Qper[i,j] = Vw[i,j] * 1/s.k[j]
        
        # 3)
        # most upper bucket
        if l==1:
            Vw[i,0] = Vw[i,0] - AET[i] - Qs[i]
        elif l==2:
            Vw[i,0] = Vw[i,0] - AET[i] - Qs[i] - Qper[i]
        else:
            Vw[i,0] = Vw[i,0] - AET[i] - Qs[i] - Qper[i,0]
            
        # check for continuity of mass
        if Vw[i,0] < 0:
            d = abs(Vw[i,0])
            Vw[i,0] = Vw[i,0] + AET[i]
            AET[i] = AET[i] - d
            if AET[i] < 0:
                AET[i] = 0
            Vw[i,0] = Vw[i,0] - AET[i]
            if Vw[i,0] < 0:
                d = abs(Vw[i,0])
                Vw[i,0] = Vw[i,0] + Qs[i]
                Qs[i] = Qs[i] - d
                Vw[i,0] = Vw[i,0] - Qs[i]
                if Vw[i,0] < 0:
                    Qs[i] = 0
                    Vw[i,0] = 0
            
        
        # lower buckets
        # for only two buckets, this does nothing
        if l > 1:
            if l == 2:
                Vw[i,1] = Vw[i-1,1] + Qper[i]
               
                 # check for continuity of mass
                if Vw[i,1] < 0:
                    d = abs(Vw[i,0])
                    Vw[i,1] = Vw[i,1] + Qper[i,2]
                    AET[i] = Qper[i,2] - d
                    if Qper[i,2] < 0:
                        Qper[i,2] = 0
                    Vw[i,1] = Vw[i,1] - Qper[i,2]
                    if Vw[i,1] < 0:
                        Qper[i,2] = 0
                        Vw[i,1] = 0
           
            elif l > 2:
                for j in range(1,l):
                    Vw[i,j] = Vw[i-1,j] + Qper[i,j] - Qper[i,j+1]
                   
                    # check for continuity of mass
                    if Vw[i,j] < 0:
                        d = abs(Vw[i,0])
                        Vw[i,j] = Vw[i,j] + Qper[i,j+1]
                        AET[i] = Qper[i,j+1] - d
                        if Qper[i,j+1] < 0:
                            Qper[i,j+1] = 0
                        Vw[i,j] = Vw[i,j] - Qper[i,j+1]
                        if Vw[i,j] < 0:
                            Qper[i,j+1] = 0
                            Vw[i,j] = 0
            
        ## for lowest bucket
        
        # release from lowest
        Qss[i] = Vw[i,-1] * 1/s.k[-1]
        
        if l == 1:
            Vw[i,0] = Vw[i,0] - Qss[i]
        elif l == 2:
            Vw[i,-1] = Vw[i,-1] - Qss[i]
        else:
            Vw[i,-1] = Vw[i,-1] - Qss[i]
        
        # check for continuity of mass
        if Vw[i,-1] < 0:
            d = abs(Vw[i,-1])
            Vw[i,-1] = Vw[i,-1] + Qss[i]
            Qss[i] = Qss[i] - d
            if Qss[i] < 0:
                Qss[i] = 0
            Vw[i,-1] = Vw[i,-1] - Qss[i]
            if Vw[i,-1] < 0:
                Qs[i] = 0
                Vw[i,-1] = 0
        
        # 4)
        # loop from lowest to most upper to check for capacitiy exceedence
        for j in reversed(range(1,l)):
            if Vw[i,j] > s.Scap[j]:
                up = Vw[i,j] - s.Scap[j]
                Vw[i,j] = s.Scap[j]
                Vw[i,j-1] = Vw[i,j-1] + up
        
        # 5) for most upper bucket
        if l == 1:
            pass
        elif Vw[i,0] > s.Scap[0]:
            q = Vw[i,0] - s.Scap[0]
            Qs[i] = Qs[i] + q
            Vw[i,0] = s.Scap[0]
                
        Q[i] = Qs[i] + Qss[i]
    
#    if any(Vw<0):
#        Warning('Negative Soil Water Storage')
    
    #%% output     
    
    # total system storage
    Vw_tot = np.sum(Vw, axis=1)
    
    data = {'Q' : Q,
            'Qs' : Qs,
            'Qss' : Qss,
            'Vw' : Vw_tot,
            'snow' : sdepth,
            'snowacc' : dsdepth,
            'PET' : PET,
            'AET' : AET,
            'Pr' : Pr,
            'Ta' : Ta
            }
    
    # single bucket storage
    i = 0
    for col in Vw.T:
        name = 'Vw%i'%i
        data[name] = col
        i += 1
    
    hyd = pd.DataFrame(data=data, index=index)
    
    return hyd


#%% RANDOM LANDSLIDES FROM HEAVY-TAILED DISTRIBUTION
#def randht(n,**kwargs):
def randht(n, *varargin, seed='none'):
    '''
    RANDHT generates n observations distributed as some continous heavy-tailed distribution. Options are power law, log-normal, stretched           exponential, power law with cutoff, and exponential. Can specify lower cutoff, if desired.
    ---------------------
    
    Input
    -----
    n : generate n observations
    *args:
        xmin : 
        Type : type of distribution as string
            - PL : pwerlaw, reqires alpha
            - PC : cutoff, requires alpha and Lambda (?)
            - EX : exponential, requires Lambda
            - LN : log-normal, requires mu and sigma
            - ST : stretched, requires Lambda and beta
            
    seed : initialize random number generator; for reproducibility; by default it is 'none', which means new randomness each time, else set a number.
    
    Output
    ------
    x : 
    
    Details
    -------
    original source : http://www.santafe.edu/~aaronc/powerlaws/
    Ported to python by Joel Ornstein (2011 August), joel_ornstein@hmc.edu
    '''
    #%% update variables
#    v = {'Type': '',
#         'xmin': 1,
#         'alpha': 2.5,
#         'beta': 1,
#         'Lambda': 1,
#         'mu': 1,
#         'sigma': 1}
#    
#    for key, value in kwargs.items():
#        if key not in v.keys():
#            raise NameError('Invalid keyword argument input.')
#            
#    v.update(kwargs)
#    globals().update(v)
#
#
#    if n<1:
#        raise AttributeError('(RANDHT) Error: invalid ''n'' argument.')
#
#    if xmin < 1:
#        raise AttributeError('(RANDHT) Error: invalid ''xmin'' argument.')

    Type   = '';
    xmin   = 1;
    alpha  = 2.5;
    beta   = 1;
    Lambda = 1;
    mu     = 1;
    sigma  = 1;


    # parse command-line parameters; trap for bad input
    i=0; 
    while i<len(varargin): 
        argok = 1; 
        if type(varargin[i])==str: 
            if varargin[i] == 'xmin':
                xmin = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'powerlaw':
                Type = 'PL'
                alpha  = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'cutoff':
                Type = 'PC';
                alpha  = varargin[i+1]
                Lambda = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'exponential':
                Type = 'EX'
                Lambda = varargin[i+1]
                i = i + 1
            elif varargin[i] == 'lognormal':
                Type = 'LN';
                mu = varargin[i+1]
                sigma = varargin[i+2]
                i = i + 2
            elif varargin[i] == 'stretched':
                Type = 'ST'
                Lambda = varargin[i+1]
                beta = varargin[i+2]
                i = i + 2
            else:
                argok=0
        
      
        if not argok: 
            print('(RANDHT) Ignoring invalid argument') #' ,i+1 
      
        i = i+1 

    if n<1:
        print('(RANDHT) Error: invalid ''n'' argument; using default.\n')
        n = 10000;

    if xmin < 1:
        print('(RANDHT) Error: invalid ''xmin'' argument; using default.\n')
        xmin = 1;


    #%% methods
    
    random.seed(seed)   #### SET THE SEED
    
    x=[]
    if Type == 'EX':
        x=[]
        for i in range(n):
            x.append(xmin - (1./Lambda)*math.log(1-random.random()))
    elif Type == 'LN':
        y=[]
        for i in range(10*n):
            y.append(math.exp(mu+sigma*random.normalvariate(0,1)))

        while True:
            y= filter(lambda X:X>=xmin,y)
            q = len(y)-n;
            if np.isclose(q, 0.):
                break

            if q>0.:
                r = range(len(y));
                random.shuffle(r)
                ytemp = []
                for j in range(len(y)):
                    if j not in r[0:q]:
                        ytemp.append(y[j])
                y=ytemp
                break
            if (q<0.):
                for j in range(10*n):
                    y.append(math.exp(mu+sigma*random.normalvariate(0,1)))
            
        x = y
        
    elif Type =='ST':
        x=[]
        for i in range(n):
            x.append(pow(pow(xmin,beta) - (1./Lambda)*math.log(1.-random.random()),(1./beta)))
    elif Type == 'PC':
        
        x = []
        y=[]
        for i in range(10*n):
            y.append(xmin - (1./Lambda)*math.log(1.-random.random()))
        while True:
            ytemp=[]
            for i in range(10*n):
                if random()<pow(y[i]/float(xmin),-alpha):ytemp.append(y[i])
            y = ytemp
            x = x+y
            q = len(x)-n
            if np.isclose(q, 0.):
                break

            if (q>0):
                r = range(len(x))
                random.shuffle(r)

                xtemp = []
                for j in range(len(x)):
                    if j not in r[0:q]:
                        xtemp.append(x[j])
                x=xtemp
                break;
            
            if (q<0.):
                y=[]
                for j in range(10*n):
                    y.append(xmin - (1./Lambda)*math.log(1.-random.random()))


    else:
        x=[]
        for i in range(n):
            x.append(xmin*pow(1.-random.random(),-1./(alpha-1.))) # random.random() is uniform distribution [0,1]

    return x
###############################################################################

#%% PROBABILISTIC HILLSLOPE EROSION
def large_ls(T, Pr, snow, Tsd, Tpr, Tsa, xmin, alpha, cutoff, Tfreeze, LStrig, area = 10.**6, seed='none'):
    '''
    Generation of large landslides by thermal trigger (procedure 1 in Bennett et al., 2014).
    Parameters taken from Bennett et al. (2012/13) are not altered.
    ---------------------
    
    Input
    -----
    T : Temperature [degreeC]
    Pr : Precipitation [mm]
    snow : data frame from degree-day-model [mm SWE]
    Tsd : threshold snowdepth for landslides to be triggered [mm SWE]
    Tpr : threshold liquid precipitation for landslides to be triggered [mm]
    Tsa : Snow temperature accumulation threshold [°C]
    xmin : 
    alpha : 
    cutoff : 
    Tfreeze : 
    LStirg : Landslide triggering mechanism ['thermal', 'rainfall', 'random']
    area : catchment area [km2], default is 10^6, i.e. output can also be interpreted in [m3]
    seed : initialize random number generator
    
    Output
    ------ 
    lrg_ls : time series of large land slides, [m3] if area not provided, else [mm]

    '''
    # parameters for large landslides distribution, from Bennett et al. (2012)
    
    if (LStrig == 'thermal') or (LStrig == 'random'):
        # T and snow have to be resampled to daily mean
        T_day = T.resample('24h').mean()
        #T_day_1 = T_day.shift(1) # this is T of 1 day before
        idx = T_day.index
        T_day = T_day.values
        #T_day_1 = T_day_1.values
        snow_day = snow.resample('24h').mean()
        snow_day = snow_day.values
        
        # a LS is triggered when T is subfreezing, the day before was not freezing and the snow depth is below threshold
        cond1 = T_day < Tfreeze                       # freezing days
        #cond2 = T_day_1 > 0                    # positive T days
        cond3 = snow_day < Tsd                  # days of only little snow
        #lsdays = cond1 & cond2 & cond3          # boolean array with days of possible landslides
        lsdays = cond1 & cond3
        N = len(lsdays[lsdays == True])         # number of big lansdslides
        
        lrg_ls = np.zeros(len(T_day))
    
    if LStrig == 'rainfall':
        Prl = Pr.copy()
        Prl[T<=Tsa] = 0 # liquid precipitation
        Prl_day = Prl.resample('24h').sum() # daily sums
        idx = Prl_day.index
        lsdays = Prl_day > Tpr
        N = len(lsdays[lsdays == True])
        
        lrg_ls = np.zeros(len(Prl_day))
    
    if LStrig == 'random':
        # N from thermal triggering
        nt = len(T_day)
        dt = int(nt/N)                       # mean (?) spacing between small landslides
        dtexp = np.random.exponential(dt, N)
        dtexp = np.ceil(dtexp)                  # get full days, in this case max one per day
        ids = np.cumsum(dtexp)
        if max(ids) >= nt:
            nids = ids/max(ids) * (nt-2) # if the days go beyond the time series length, rescale to length of time series // (t-2) to ensure max value after ceiling in next line as well
            nids = np.ceil(nids)
            ids = nids
        ids = [int(i) for i in ids]
        lsdays = ids
        
    # generate N large landslide magnitudes (volume, m3). iteration is needed in order to avoid unreasonable large volumes, greater than cutoff
    # this is not effective computation...condition should be in randth
    cond=False
    while not cond:
        mags = randht(N,'xmin',xmin,'powerlaw', alpha, seed=seed) 
        cond = max(mags)<cutoff
        if not cond:
            seed = seed + 10000
    
    #output
    lrg_ls[lsdays] = mags
    lrg_ls = lrg_ls / area * 10.**-3        # convert m3 to mm
    data = {'mag': lrg_ls}
    lrgls = pd.DataFrame(data, index=idx)
    
    return lrgls

def small_ls(t, N, xmin, area=10.**6, seed=None):
    '''
    Generation of small landslides (procedure 1 in Bennett et al., 2014).
    Parameters taken from Bennett et al. (2012/13) are not altered.
    ---------------------
    
    Input
    -----
    t : length of time series, number of days as integer
    N : number of large landslides, because the number of small landslides comes from a ratio
    xmin : Minimum landslide volume from the power-law tail
    area : catchment area [km2], default is 10^6, i.e. output can also be interpreted in [m3]
    seed : initilaize random state of the generator, defualt=None, else set a number
    
    Output
    ------ 
    sls : time series of small land slides, [m3] if area not provided, else [mm]
    '''
    
    np.random.seed(seed)
    
    # Parameters of lognormal distribution fit to landslides <xmin, from Bennett et al. (2012)
    mu=3.36                 # Mean of lognormal distribution
    sigma=1.18              # Standard deviation of lognormal distribution

    # Other parameters
    ratio = 3.36            # 3.36 is the average ratio of small to large failures
    
    # generate spacing of small LS
    s_ls = np.zeros(t)                  # initialize days
    n = int(ratio * N)                  # number of small LS according to ratio of small/large LS
    dt = int(t/n)                       # mean (?) spacing between small landslides
    dtexp = np.random.exponential(dt, n)
    dtexp = np.ceil(dtexp)                  # get full days, in this case max one per day
    ids = np.cumsum(dtexp)
    if max(ids) >= t:
        nids = ids/max(ids) * (t-2) # if the days go beyond the time series length, rescale to length of time series // (t-2) to ensure max value after ceiling in next line as well
        nids = np.ceil(nids)
        ids = nids
        
    # generate small landslides
    mags_teo = np.random.lognormal(mu, sigma, 10**6)    # draw many from random distribution, should represent the theoretical distribution
    mags_con = mags_teo[mags_teo <= xmin]               # represents theoretical distribution but constrained by xmin
    mags = np.random.choice(mags_con,n)                 # n samples from constrained distribution
    
    #output
    ids = [int(i) for i in ids]
    s_ls[ids] = mags
    s_ls = s_ls / area * 10.**-3                        # convert m3 to mm
    
    data = {'mag': s_ls}
    sls = pd.DataFrame(data=data)
    
    return sls

#%% SEDIMENT TRANSFER MODEL

def sedcas(Lls, Sls, hyd, qdf, smax, rhc, shcap, area, method, LStrig, Tpr, shinit=0, scinit=0, **kwargs):
    '''
    Sediment cascade from hillslope to channel to outlet.
    Note: inactive storage not considered yet, but it's not really needed...
    ---------------------
    
    Input
    -----
    Lls : time series of sum of large landslides
    Sls : time series of sum of small landslides
    q : discharge from hydro model [mm]
    snow : snow depth SWE [mm] from hydro model
    smax : maximim potential volumetric ration of sediment (with density of bedrock) to water in the flow
    rhc : redopsition rate from hillslope to channel [-]
    shcap : hillslope storage capacity [mm]
    area : catchment area [km2]
    method : method for sediment transport, ['lin' 'exp']
    qdf : critical discharge for triggering of debris flow [mm/t?]
    shinit : initial hillslope storage [mm]
    scinit : initial channel storage [mm]
    kwargs : depending on the method
        if method is 'lin' semdiment transport starts when a critical discharge is exceeded:
            no additional inputs required (only qdf)
        if method is 'exp' sediment transport is follows discharge in an exponential relationship:
            (a : scaling parameter , determined automatically)
            b : shape parameter
        mindf : if given, sediment output will also be given in terms of debris flows [mm]
        smax_nodf : max sediement concentration for sub-critical flow conditions
    
    Output
    ------ 
    sed : data frame containing...
        sh : hillslope storage time series [mm]
        sc : channel storage time series [mm]
        so : catchment sediment output time series [mm]
        sopot : potential sediment output based on discharge [mm]
        dfs : debris flows, sediment output above minimum threshold and concentration of debris flows[mm]
        conc : sediment concentration in flow [-]
    '''
    q = hyd.Qs.copy()
    snow = hyd.snow.copy()
    
    # check if given kwargs are valid
    valid_kwargs = ['b','mindf', 'smax_nodf']
    for key in kwargs.keys():
        if not key in valid_kwargs:
            raise AttributeError('%s is not a valid property.'%key)
            
    # check if required arguments are povided, else raise error
    if method == 'exp':
        if not ('a' and 'b' and 'smax_nodf' and 'mindf') in kwargs.keys():
            raise AttributeError('for method "exp" keyword arguments a and b must be provided.')
    
    # unpack kwargs
    def unpack_kwargs(kwargs, key):
        try:
            var = kwargs[key]
        except KeyError:
            var = np.nan
        return var
    b = unpack_kwargs(kwargs, 'b')
    mindf = unpack_kwargs(kwargs, 'mindf')
    smax_nodf = unpack_kwargs(kwargs, 'smax_nodf')
            
    # determine 'a' and 'Qmin_nondf'
        # this is based on two facts
            # 1) the sediment concentration for sub-critical bedload transport cannot exceed the concentration given by smax_nodf
            # 2) the volume of the sediment transported cannot exceed the minimal debris-flow solid volume
    
    if method == 'exp':
        Qmin_nodf = qdf - (mindf*(1-smax_nodf) / (smax_nodf*qdf))**(1/(1-b))
        if Qmin_nodf < 0:
            Qmin_nodf = 0
        a = smax_nodf * qdf / ((qdf-Qmin_nodf)**b*(1-smax_nodf))
    
    # landslides (ls) are daily, needs to be padded
    freq = q.index[1] - q.index[0]             # desired frequency
    delta = pd.to_timedelta('1 day') - freq
    dates = Lls.index.values.copy()
    dates[-1] = dates[-1] + delta
    if 'datetime' in str(Lls.index.dtype):
        dates = pd.to_datetime(dates)    
    elif 'timedelta' in str(Lls.index.dtype):
        dates = pd.to_timedelta(dates, unit='h')
    else:
        raise AttributeError('Your input index must be of type "timedelta" or "datetime"')  
    
    if (LStrig == 'thermal') or (LStrig == 'random'):
        ls = Lls.mag.copy() + Sls.mag.copy()
        ls.index = dates                      # the last value is now the same as for the sub-daily time-series, needed for padding
        ls = ls.resample(freq).pad()               # this is the time series at desired frequency
        
        if 'datetime' in str(ls.index.dtype):
            cond = ls.index.time == pd.to_datetime('12:00').time()     # hillslope failure always happen at noon
            ls[~cond] = 0                                              # set the other hours to 0
        elif 'timedelta' in str(ls.index.dtype):
            cond1 = ls.index.astype('timedelta64[h]').astype('int64') % 12 == 0
            cond2 = ls.index.astype('timedelta64[h]').astype('int64')/12 % 2 == 1
            ls[~(cond1 & cond2)] = 0

    elif LStrig == 'rainfall':
        ### NOT SURE THIS IS GENERIC FOR ALL TEMPORAL RESOLUTIONS AND DATA TYPES
        Prc = hyd.Pr.groupby(pd.Grouper(freq='24h')).cumsum() # daily cumsum
        Prc[Prc <= Tpr] = np.nan    # set all smaller than the triggering threshold to nan
        
        daily_min = Prc.groupby(pd.Grouper(freq='24h')).min() # this is the minimum of each day
        daily_min.index = dates
        daily_min = daily_min.resample(freq).pad() # assign min for every modeling time step
        
        diff = Prc - daily_min
        cond1 = diff == 0            # where the difference to the min is 0, is the first time the precipitation exceeds the triggering threshold on that day
        dfcond = pd.DataFrame(data=cond1)
        dfcond.columns = ['cond1']
        dfcond['cond2'] = np.nan
        dfcond['cond2'][1:] = dfcond.cond1[:-1]
        dfcond.cond2.iloc[0] = False
        cond1 = dfcond.cond1.values
        cond2 = dfcond.cond2.values
        
        Lls.index = dates
        Lls = Lls.resample(freq).pad()
        Lls[~cond1] = 0 # where the difference is not 0
        Lls[cond2] = 0 # where the the difference is 0, but the previous one is already tagged
        
        Sls.index = dates
        Sls = Sls.resample(freq).pad()
        if 'datetime' in Sls.index.dtype_str:
            cond = Sls.index.time == pd.to_datetime('12:00').time()     # hillslope failure always happen at noon
            Sls[~cond] = 0                                              # set the other hours to 0
        elif 'timedelta' in Sls.index.dtype_str:
            cond1 = Sls.index.astype('timedelta64[h]').astype('int64') % 12 == 0
            cond2 = Sls.index.astype('timedelta64[h]').astype('int64')/12 % 2 == 1
            Sls[~(cond1 & cond2)] = 0
        
        ls = Lls.mag.copy() + Sls.mag.copy()
    
    # test if too long
    i = np.argwhere(ls.index == q.index[-1])[0][0]
    ls = ls[:i+1]
    
    # convert to arrays if Data Frames or Series
    try:
        idx = ls.index
        ls = ls.values
        q = q.values
        snow = snow.values
    except AttributeError:
        pass
    
    # initialize    
    t = len(q)      # length of time series
    sh, sc, so, sopot = np.zeros(t), np.zeros(t), np.zeros(t), np.zeros(t)      # initialize output arrays
    
    # initial conditions
    sh[0] = shinit
    sc[0] = scinit
    
    # parameters for large landslides distribution, from Bennett et al. (2012)
    xmin=233                # Minimum landslide volume from the power-law tail
    alpha=1.65              # Power law scaling exponent in landslide distribution
    
    for i in range(1,t):
        
        ## TRANSFER NUMERICAL SCHEME: outputs at t depend on storage states at t, i.e. first the storages, considering inputs, are recomputed, then the debris flow triggering is computed.
        
        # hillslope-channel transfer
        # concept: from each landslide a fraction defined by the rhc parameter is redeposited in the channel storage, the rest of the landslide goes directly into the channel
        redep = ls[i]*rhc                        # redeposition according to redeposition rate
        sh[i] = sh[i-1] + redep                  # hillslope storage change
        sc[i] = sc[i-1] + ls[i] - redep          # channel storage change
        if sh[i] > shcap:
            ls_sh = shcap*2
            while ls_sh >= shcap:
                ls_sh = randht(1,'xmin',xmin,'powerlaw',alpha)[0]
                ls_sh = ls_sh / area * 10.**-3      # convert m3 to mm
            sh[i] = sh[i] - ls_sh
            sc[i] = sc[i] + ls_sh
        
        # channel-outlet transfer
        # # there are two methods for sediment entrainment: lin and exp...
        # # Note
        # # qdf is equivalent to the critical shear stress
        # # q-qdf is equivalent to the excess shear stress
        if method == 'lin':
            # #(1) If runoff exceeds a threshold, Qdf, a debris flow is triggered
            if q[i] >= qdf:
                sopot[i] = smax/(1-smax) * q[i]# - qdf)         # potential sediment output
            
        if method == 'exp':
            if (q[i] < qdf) and (q[i] >= Qmin_nodf):
                sopot[i] = a * (q[i] - Qmin_nodf) ** b                        # exponential function for small flows
            elif q[i] >= qdf:
                sopot[i] = smax/(1-smax) * q[i]                          # linear function for debris flows (flows above qdf)
        
        # # the sediment flow might, however, not be initiated because...
        # case 1: there is snow --> no DF
        if snow[i] > 0:
            so[i] = 0
            sopot[i] = 0
        # case 2: transport limited, channel storage is big enough
        elif sc[i] >= sopot[i]:
            so[i] = sopot[i]
            sc[i] = sc[i] - so[i]
        # case 3: supply limited, channel storage is too small
        else:
            so[i] = sc[i]
            sc[i] = 0
    
    ############
        
    # output is debris flow when 1) the volume is larger than mindf and 2) the sediment concentration is larger than smax_nodf
    if ('mindf') in kwargs.keys():
        
        def get_dfs(q,s,mindf):
            q2 = q.copy()
            q2[q2==0] = np.nan
            conc = s/(s+q2)             # volumetric sediment concentration in flow
            cond1 = s >= mindf         # first condition, sediment output must be greater than the minimum possibld DF
            cond2 = conc > smax_nodf    # second condition, the sediment concentration must be greater than for fluvial transport
            dfs = s[cond1 & cond2]
            idxdfs = idx[cond1 & cond2]
            dt = idxdfs[1:] - idxdfs[:-1]   # get spacing between debris flows
            dt = dt.insert(0,pd.NaT)        # put a NaT at the first position becuase  it doesn't have a valiue before
            
            # if there are consecutive values, add them
            dfsnew = dfs.copy()
            for i in range(len(dt)-1,0,-1):
                if dt[i] == pd.to_timedelta('1 hour'):   # this should mean, that when using daily data, the values are not added
                    dfsnew[i-1] = dfsnew[i-1] + dfsnew[i]   # add to the previous hour
                    dfsnew[i] = 0
            idxdfs = idxdfs[dfsnew>0]
            dfsnew = dfsnew[dfsnew>0]
            
            # insert values in full modelling time series
            dfs = pd.Series(np.zeros(len(idx)), index=idx)
            dfs.loc[idxdfs] = dfsnew
            dfs = dfs.values
            
            return dfs, conc
        
        df, conc = get_dfs(q,so,mindf)
        dfp, concp = get_dfs(q,sopot,mindf)
                
    # output    
    data = {'sh': sh,
            'sc': sc,
            'so': so,
            'sopot': sopot,
            'ls': ls,
            'dfs': df,
            'conc': conc,
            'dfspot': dfp,
            'concpot': concp}
    
    sed = pd.DataFrame(data=data, index=idx)
    
    return  sed
