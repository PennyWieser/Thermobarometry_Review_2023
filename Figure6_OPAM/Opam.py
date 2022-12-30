import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Thermobar
import scipy
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

## Alternative OPAM formulations

def Herzberg2004(Liq_Comps):
    '''
    Method described in Herzberg et al. (2004) to calculate the pressure of OPAM-saturated liquids.

    Parameters:
    ----------
    Liq_Comps: Pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'

    Returns:
    ----------
    liq_Comps: Pandas DataFrame
        Input DataFrame with a column added displaying the calculated pressure in kbar.

    '''
    mol_perc = 100*Thermobar.calculate_anhydrous_mol_fractions_liquid(Liq_Comps)

    # calculate projections
    An = mol_perc['Al2O3_Liq_mol_frac'] + mol_perc['Cr2O3_Liq_mol_frac'] + mol_perc['TiO2_Liq_mol_frac']
    Di = mol_perc['CaO_Liq_mol_frac'] + mol_perc['Na2O_Liq_mol_frac'] + 3*mol_perc['K2O_Liq_mol_frac'] - mol_perc['Al2O3_Liq_mol_frac'] - mol_perc['Cr2O3_Liq_mol_frac']
    En = mol_perc['SiO2_Liq_mol_frac'] - 0.5*mol_perc['Al2O3_Liq_mol_frac'] - 0.5*mol_perc['Cr2O3_Liq_mol_frac'] - 0.5*mol_perc['FeOt_Liq_mol_frac'] - 0.5*mol_perc['MnO_Liq_mol_frac'] - 0.5*mol_perc['MgO_Liq_mol_frac'] - 1.5*mol_perc['CaO_Liq_mol_frac'] - 3*mol_perc['Na2O_Liq_mol_frac'] - 3*mol_perc['K2O_Liq_mol_frac']

    # ensure projections sum to 100
    Bottom = An + Di + En
    An = 100*An/Bottom
    Di = 100*Di/Bottom
    En = 100*En/Bottom

    Liq_Comps['P_kbar_calc'] =10 * (-9.1168 + 0.2446*(0.4645*En + An) - 0.001368*((0.4645*En + An)**2))

    return Liq_Comps

def Villiger2007(Liq_Comps):
    '''
    Method described in Villiger et al. (2007) to calculate the pressure of OPAM-saturated liquids.

    Parameters:
    ----------
    Liq_Comps: Pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'

    Returns:
    ----------
    liq_Comps: Pandas DataFrame
        Input DataFrame with a column added displaying the calculated pressure in kbar.

    '''
    MgNo = (Liq_Comps['MgO_Liq']/40.3044)/(Liq_Comps['MgO_Liq']/40.3044 + Liq_Comps['FeOt_Liq']/71.844)
    P_kbar = (Liq_Comps['CaO_Liq'] - 3.98 - 14.96*MgNo)/(-0.260)

    Liq_Comps['P_kbar_calc'] = P_kbar

    return Liq_Comps


## Equations for XAl, XMg, and XCa

def Voigt2017(Liq_Comps, P):
    '''
    Equations for the cation fraction of Al, Mg, and Ca in basaltic melts from Voigt et al. (2017).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    XAl, XMg, XCa: numpy array
       Predicted cation fractions of Al, Mg, and Ca

    '''
    XAl=0.239+0.01801*P+0.162*Liq_Comps['Na_Liq_cat_frac']+0.485*Liq_Comps['K_Liq_cat_frac']-0.304*Liq_Comps['Ti_Liq_cat_frac']-0.32*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.353*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']-0.13*Liq_Comps['Si_Liq_cat_frac']+5.652*Liq_Comps['Cr_Liq_cat_frac']
    XCa=1.07-0.02707*P-0.634*Liq_Comps['Na_Liq_cat_frac']-0.618*Liq_Comps['K_Liq_cat_frac']-0.515*Liq_Comps['Ti_Liq_cat_frac']-0.188*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.597*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']-3.044*Liq_Comps['Si_Liq_cat_frac']-9.367*Liq_Comps['Cr_Liq_cat_frac']+2.477*(Liq_Comps['Si_Liq_cat_frac']**2)
    XMg=-0.173+0.00625*P-0.541*Liq_Comps['Na_Liq_cat_frac']-1.05*Liq_Comps['K_Liq_cat_frac']-0.182*Liq_Comps['Ti_Liq_cat_frac']-0.493*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.028*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']+1.599*Liq_Comps['Si_Liq_cat_frac']+3.246*Liq_Comps['Cr_Liq_cat_frac']-1.873*(Liq_Comps['Si_Liq_cat_frac']**2)

    return XAl, XMg, XCa

def Yang1996(Liq_Comps, P):
    '''
    Equations for the cation fraction of Al, Mg, and Ca in basaltic melts from Yang et al. (1996).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    XAl, XMg, XCa: numpy array
       Predicted cation fractions of Al, Mg, and Ca
    '''
    XAl=0.236+0.00218*(P*10)+0.109*Liq_Comps['Na_Liq_cat_frac']+0.593*Liq_Comps['K_Liq_cat_frac']-0.350*Liq_Comps['Ti_Liq_cat_frac']-0.299*Liq_Comps['Fet_Liq_cat_frac']-0.13*Liq_Comps['Si_Liq_cat_frac']
    XCa=1.133-0.00339*(P*10)-0.569*Liq_Comps['Na_Liq_cat_frac']-0.776*Liq_Comps['K_Liq_cat_frac']-0.672*Liq_Comps['Ti_Liq_cat_frac']-0.214*Liq_Comps['Fet_Liq_cat_frac']-3.355*Liq_Comps['Si_Liq_cat_frac']+2.830*(Liq_Comps['Si_Liq_cat_frac']**2)
    XMg=-0.277+0.00114*(P*10)-0.543*Liq_Comps['Na_Liq_cat_frac']-0.947*Liq_Comps['K_Liq_cat_frac']-0.117*Liq_Comps['Ti_Liq_cat_frac']-0.490*Liq_Comps['Fet_Liq_cat_frac']+2.086*Liq_Comps['Si_Liq_cat_frac']-2.4*(Liq_Comps['Si_Liq_cat_frac']**2)

    return XAl, XMg, XCa

## minimisation calculation

def findMin(P_Gpa, liq_cat, equationP):
    '''
    Funcion used to identify the location of the minimum value of X2 (modified Chi-squared), which is a function of pressure. Individually, the function calculates the modified Chi-squared value for a given pressure and melt composition, but when combined with the scipy.optimize.minimize_scalar function this can be used to determine the pressure at which the minimum value of X2 is located.

    Parameters:
    -----------
    P_GPa: float
        Pressure (in GPa).

    liq_cat: dict
        Dictionary containing the melt cation fractions.

    equationP: string
        Specifies the equations used to calculate the theorectical cation fractions of Al, Mg, and Ca at different pressures. Choice of "Yang1996" or "Voigt2017". "Yang1996" is used as default.

    Returns:
    ----------
    X2: float
        Value of the modified Chi-Squared expression from Hartley et al. (2018) at the specified pressure.
    '''

    if equationP == "Yang1996":
        XAl, XMg, XCa = Yang1996(liq_cat, P_Gpa)
    else:
        XAl, XMg, XCa = Voigt2017(liq_cat, P_Gpa)

    X2 = ((liq_cat['Al_Liq_cat_frac']-XAl)/(0.05*liq_cat['Al_Liq_cat_frac']))**2+((liq_cat['Ca_Liq_cat_frac']-XCa)/(0.05*liq_cat['Ca_Liq_cat_frac']))**2+((liq_cat['Mg_Liq_cat_frac']-XMg)/(0.05*liq_cat['Mg_Liq_cat_frac']))**2

    return X2

## OPAM calculations

def pyOpam(*, liq_comps = None, equationP = None, fo2 = None, equationT = None):
    '''
    Calculate multi-phase saturation pressure and 'probability' of three phase saturation for basaltic melts. Pressure sensitive equations for the cation fraction of Al, Mg, and Ca in the melt phase are used to determine the pressure of storage by identifying the location of the minimum misfit between the observed and calculated cation fractions.

    Parameters:
    -----------
    liq_comps: pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'.

    equationP: string
        Specifies the equations used to calculate the theorectical cation fractions of Al, Mg, and Ca at different pressures. Choice of "Yang1996" or "Voigt2017". "Yang1996" is used as default.

    fo2: string
        fO2 buffer to use if the equations of Voigt et al. 2017 are chosen and no Fe3Fet_Liq is specified in the liq_comps dataframe. Here the buffer will be used to calculate the Fe3Fet_Liq value for each composition in the dataframe using Thermobar and a Temperature specified by "equationT"

    equationT: string
        If fo2 is not None, a liquid-only thermometer from Thermobar can be specified here to calculate the temperature of the different liquids in liq_comps prior to calculation of the Fe3Fet_Liq value for each sample. As default, "T_Helz1987_MgO" is used.

    Returns:
    ----------
    liq_comp: pandas DataFrame
        copy of the initial DataFrame with two columns added: calculated pressures and 'probability' of three-phase saturation
    '''

    # check all required parameters are present
    if liq_comps is None:
        raise Exception("No composition specified")

    if equationP is None:
        equationP = "Yang1996"

    if fo2 is None and equationP == "Voigt2017":
        raise Warning(f'{equationP} requires you to specify the redox state of the magma. Please ensure the Fe3+/Fe_total ratio of the melt phase is specified in the input dataframe.')

    # create a copy of the input dataframe
    liq_comp = liq_comps.copy()
    liq_comp = liq_comp.reset_index(drop = True)

    # determine Fe3 from fO2 if set
    if fo2 is not None:
        if equationT is None:
            equationT="T_Helz1987_MgO"

        T = Thermobar.calculate_liq_only_temp(liq_comps=liq_comp,  equationT=equationT)-273.15

        liq_comp_Fe3 = Thermobar.convert_fo2_to_fe_partition(liq_comps=liq_comp, T_K=T+273.15, P_kbar=3, fo2=fo2, model="Kress1991", renorm=False)

        liq_comp['Fe3Fet_Liq'] = liq_comp_Fe3['Fe3Fet_Liq']

    #calculatoin cation fractions and create nparrays to store outputs
    liq_cats=Thermobar.calculate_anhydrous_cat_fractions_liquid(liq_comp)
    P = np.zeros(len(liq_cats['SiO2_Liq']))
    Prob = np.zeros(len(liq_cats['SiO2_Liq']))

    # minimise P for each sample and calculate Pf using scipy.stats.chi2
    for t in range(len(liq_cats['SiO2_Liq'])):
        liq_cat = liq_cats.loc[t].to_dict().copy()
        res = minimize_scalar(findMin, method = 'brent', args = (liq_cat, equationP))
        P[t] = res.x
        Stats = res.fun
        Prob[t] = 1 - chi2.cdf(Stats, 2)

    # save outputs into the dataframe, converting P into kbar
    liq_comp['P_kbar_calc'] = P*10
    liq_comp['Pf'] = Prob

    return liq_comp