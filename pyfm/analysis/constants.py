"""
Constants employed in g-2 analysis
"""
import numpy as np

MPI_ISO = 135/1000.0
MK_ISO = 494.6/1000.0
MD_ISO = 1967/1000.0

#MPI_ISO = 134.997/1000.0
#MK_ISO = 494.496/1000.0
#MD_ISO = 1967.02/1000.0

pi = np.pi

riemanZeta3 = 1.2020569031595942853997381615114499907649862923404988817922715553

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421

APERY=1.20205690315959428539973816151144999

#FM_TO_GEV_INV =0.1973269718 ##from PDG 
FM_TO_GEV_INV =0.197326963 ##from PDG 

## QED coupling constant
ALPHA = 1/137.035999679

## Strong coupling at Mz

ALPHA_S_Mz = .1180

## Muon mass
#MMUON_GEV = 0.1056583715 ##GeV from PDG
MMUON_GEV = 0.10565838 ##GeV from PDG


# Quark charges
QU = +2.0
QD = -1.0
QS = -1.0
QC = +2.0

## Used in a^2 term in continuum extrapolation
LAMBDAQCD_GEV = 0.5       ##in GeV

Lambda_MS_1 = 0.48
Lambda_MS_2 = 0.48


## Meson masses
MPI0_GEV = 134.9766/1000.0   ##in GeV      
MPI0_ERR_GEV = 0.0006/1000.0 ##from PDG

MPIPLUS_GEV = 139.57018/1000.0     ##in GeV
MPIPLUS_ERR_GEV = 0.00035/1000.0   ##from PDG

MRHO_GEV = 0.77526           ##in GeV
MRHO_ERR_GEV = 0.00025       ##from PDG

MRHO_BARE_GEV = 0.766        #Chiral Model

GAMMARHO_GEV = 0.1474          ##in GeV
GAMMARHO_ERR_GEV = 0.0008      ##from PDG

## Decay constants
FRHO_PHYS_GEV = 0.21         ##HPQCD 1601.03071
FRHO_PHYS_ERR_GEV = 0.01     ##See footnote 2

FPI_PDG_GEV = 130.2/1000.0

GRHO=5.4  #Chiral Model
GRHOPIPI=6.0 #Chiral Model

F=92.21 /1000 #F from ChpT in gev

l6 = 16  #low energy constant in ChpT
