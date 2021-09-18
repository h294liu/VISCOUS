#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:31:29 2021

@author: hongliliu
"""
import numpy as np
import pandas as pd
import os, sys, argparse 
import random
random.seed(0)

sys.path.append('../')
import functions.viscous as vs


def process_command_line():
    '''Parse the commandline'''
    parser = argparse.ArgumentParser(description='Script to icalculate model evaluation statistics.')
    parser.add_argument('nSample', help='total number of samples as input.')
    args = parser.parse_args()
    return(args)

def Ishigami(x1,x2,x3):
    # reference: https://uqworld.org/t/ishigami-function/55
    f = 0
    a = 7 
    b = 0.05
    f = f + np.sin(x1)
    f = f + a * np.power(np.sin(x2), 2)
    f = f + b * np.power(x3, 4) * np.sin(x1)
    return f

def save_output(varNames,sArr,sName,ofile):
    values = list(zip(varNames,sArr))
    df = pd.DataFrame(values, columns=['Variable', sName])
    df.to_csv(ofile, header=None, index=None, sep='\t', mode='w',float_format='%.3f')
    print(df)
    return

if __name__ == "__main__":

    # process command line 
    # check args
    if len(sys.argv) != 2:
        print("Usage: %s <nSample>" % sys.argv[0])
        sys.exit(0)
    # otherwise continue
    args = process_command_line()    
    nSample = int(args.nSample)

    # other configurations
    nVar = 3
    paramLowerLimit = (-1)*np.pi
    paramUpperLimit = np.pi
            
    # ============================================================
    # Generate param samples and corresponding responses. 
    paramSamples = np.zeros((nSample,nVar))
    for iVar in range(nVar):
        paramSamples[:,iVar] = np.random.uniform(low=paramLowerLimit, high=paramUpperLimit, size=nSample)
    responseSamples = Ishigami(paramSamples[:,0],paramSamples[:,1],paramSamples[:,2]) # shape(nSample,)   
    if len(np.shape(responseSamples)) == 1:
        responseSamples = np.reshape(responseSamples,(len(responseSamples),1))

    # Save samples and response
    samples = np.concatenate((paramSamples,responseSamples), axis=1)
    np.savetxt('samples.txt',samples,delimiter=',',header='X1,X2,X3,Y')
            
    # ============================================================
    # Prepare for sensitivity analysis.
    [nSample, nVar] = np.shape(paramSamples)                  # nSample is the total number of param samples. nVar is the total number of param variables.
    GSAIndex = vs.define_GSA_param_index(nVar)                # define param groups and their corresponding index for sensitivity analysis    
    order = 1 #0                                              # 1: use linear regression to fit and predict response. 0: no regression, predict y = mean of y.
    nMC = 10000                                               # number of Monte Carlo samples.

    x = paramSamples.copy()                                   # shape(nSample,nVar)
    y = responseSamples.copy()                                # sahpe(nSample,1)
    nGSAIndex = len(GSAIndex)
    varNames = ['X1','X2','X3']

    # ============================================================
    # First-order and total-effect sensitivity analysis.
    # calculate the first-order sensitivity index
    sFirst = vs.GMCM_GSA_first_order(x,y,GSAIndex,order,nMC)  
#     np.savetxt('sFirst.txt',sFirst,delimiter=',',fmt='%.3f', header='X1,X2,X3')
    save_output(varNames,sFirst,'sFirst','sFirst.txt')
            
    # calculate the total-effect sensitivity index
    sTotal = vs.GMCM_GSA_total_effect(x,y,GSAIndex,order,nMC) 
    np.savetxt('sTotal.txt',sTotal,delimiter=',',fmt='%.3f', header='X1,X2,X3')
    save_output(varNames,sTotal,'sTotal','sTotal.txt')

    # ============================================================
    # Experiment 1: build GMM with (x,y) vs. with (x, residual).
    # calculate the first-order sensitivity index
    sFirst_residual = vs.GMCM_GSA_first_order_residual(x,y,GSAIndex,order,nMC) 
#     np.savetxt('sFirst_residual.txt',sFirst_residual,delimiter=',',fmt='%.3f', header='X1,X2,X3')
    save_output(varNames,sFirst_residual,'sFirst_residual','sFirst_residual.txt')

    # Experiment 2: calculate gmm_y_marg_pdf using the fitted GMM. vs. using the standard normal distribution.
    # calculate the first-order sensitivity index
    sFirst_normPDF = vs.GMCM_GSA_first_order_normPDF(x,y,GSAIndex,order,nMC) 
#     np.savetxt('sFirst_normPDF.txt',sFirst_normPDF,delimiter=',',fmt='%.3f', header='X1,X2,X3')
    save_output(varNames,sFirst_normPDF,'sFirst_normPDF','sFirst_normPDF.txt')

    # calculate the total-effect sensitivity index
    sTotal_normPDF = vs.GMCM_GSA_total_effect_normPDF(x,y,GSAIndex,order,nMC) 
#     np.savetxt('sTotal_normPDF.txt',sTotal_normPDF,delimiter=',',fmt='%.3f', header='X1,X2,X3')
    save_output(varNames,sTotal_normPDF,'sTotal_normPDF','sTotal_normPDF.txt')
