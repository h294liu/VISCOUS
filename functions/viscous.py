# Version V0-2021
# The algorithm is proposed by Sheikholeslami et al. (2021)
#----------------------------------------------------------------
# Programmed by Hongli Liu, University of Saskatchewan.
# E-mail: hongli.liu@usask.ca
# ----------------------------------------------------------------
# Original paper:
# Sheikholeslami, R., Gharari, S., Papalexiou, S. M., & Clark, M. P. (2021) 
# VISCOUS: A variance-based sensitivity analysis using copulas for efficient identification of dominant hydrological processes.
# Water Resources Research, 57, e2020WR028435. https://doi.org/10.1029/2020WR028435

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm, gaussian_kde
from scipy.stats import multivariate_normal
from scipy.stats.mstats import mquantiles
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt


def define_GSA_variable_index(nVar):
    """ Create variable indices for which the variance-based sensitivity indices are estimated.
    GSA: global sensitivity analysis.
    
    Parameters
    -------
    nVar: the total number of input variables (eg, parameters).
    
    Returns
    -------
    GSAIndex : the indices of variable groups to be evaluated. For example,    
    GSAIndex = {[1],[2],[3],[1,2],[1,3],[2,3],...} for group of variables.
    
    Notes
    -------
    For now only the first-order sensitivity is calculated, thus {[1],[2],[3]}.
    This code can be extended to explicitly calculate interaction effect (eg, second-order, third-order sensitivity indices)."""
    
    GSAIndex = []
    for d in range(nVar):
        GSAIndex.append([d]) # Index starts from zero by following python syntax.
    return GSAIndex

def standardize_data(data):
    """ Standradize random data into the standard normal distribution.
    referenece: https://stackoverflow.com/questions/52221829/python-how-to-get-cumulative-distribution-function-for-continuous-data-values
     
    Parameters
    -------
    data: data array to be standardized.
    
    Returns
    -------
    dataNorm: standardized data.
    
     Notes
    -------
    A faster way of calculating CDF. 
    Modify source code of kde.integrate_box_1d, and make it process array, not element-by-element. 
    reference: https://stackoverflow.com/questions/47417986/using-scipy-gaussian-kernel-density-estimation-to-calculate-cdf-inverse
    from scipy.special import ndtr
    stdev = np.sqrt(kde.covariance)[0, 0]
    pde_cdf = 1 - ndtr(np.subtract.outer(x, n)/stdev).mean(axis=1)"""    
    
    data_shape = np.shape(data)
    data = data.reshape((1,len(data)))
    
    kde = gaussian_kde(data)
    cdf_function = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    cdf = cdf_function(data)
    
    dataNorm = norm.ppf(cdf, loc=0, scale=1)
    dataNorm = dataNorm.reshape(data_shape)
    
    return dataNorm

def fit_GMM(x,y):
    """ Fit the Gaussian mixture model (GMM).
    reference: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    reference: https://cmdlinetips.com/2021/03/gaussian-mixture-models-with-scikit-learn-in-python/
    
    Parameters
    -------
    x: array, shape (n_samples, n_variables). Variable samples in normal space. 
    y: array, shape (n_samples, ). Response samples in normal space. 
        
    Returns
    -------
    best_model: object. The best fitted GMM. 
    
    Notes
    -------
    In GMM fitting, combine all the variables of x and y, and treat them as multivariates of GMM. """    
    
    # Part 1.  Fit the data with the Gaussian Mixture Model using different number of clusters.
    if len(np.shape(x)) == 1: # if shape(nSample,) -> (nSample,1)
        x = x.reshape(-1,1) 
        
    nVar = np.shape(x)[1]
    n_components = np.arange(1, nVar+20) # Note: +20 is hard coded.
    
    # Combine all the variables of x and y, and treat them as multivariates of the Gaussian mixture model.
    data = np.concatenate((x,y), axis=1)
    models = [GaussianMixture(n,covariance_type='full').fit(data) for n in n_components]
    
    # Part 2. Compute the BIC score for each model.
    gmm_model_comparisons=pd.DataFrame({"n_components" : n_components,
                                  "BIC" : [m.bic(data) for m in models]})

    # Part 3. Identify the minimum BIC score corresponding index.
    best_model_index = gmm_model_comparisons['BIC'].idxmin()
    
    # Part 4. Identify the optimal Gaussian mixture model.
    best_model = models[best_model_index]
    
    return best_model

def generate_GMM_sample(gmm, n_samples):
    """Generate random samples from the fitted Gaussian mixture model (GMM).
    reference: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/mixture/_base.py#L396
    Note this function is re-wrote here because sklearn.mixture.GaussianMixture fixes its random seed in sampling.
    
    Parameters
    -------
    gmm: object. Fitted GMM.
    n_samples : int. Number of samples to generate.
    
    Returns
    -------
    X : array, shape (n_samples, n_variables). Randomly generated sample.
    y : array, shape (nsamples,). Component labels. """

    n_samples_comp = np.random.multinomial(n_samples, gmm.weights_)

    if gmm.covariance_type == 'full':
        X = np.vstack([
            np.random.multivariate_normal(mean, covariance, int(sample))
            for (mean, covariance, sample) in zip(
                gmm.means_, gmm.covariances_, n_samples_comp)])
    y = np.concatenate([np.full(sample, j, dtype=int)
                       for j, sample in enumerate(n_samples_comp)])

    return (X, y)

def generate_data_sample(data, n_samples):

    data = data.reshape((1,len(data)))

    kde = gaussian_kde(data)
    MC_data = kde.resample(n_samples)

    cdf_function = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    MC_dataCDF = cdf_function(MC_data)
    MC_dataNorm = norm.ppf(MC_dataCDF, loc=0, scale=1)

    MC_data = MC_data.T
    MC_dataNorm = MC_dataNorm.T
    MC_dataNorm_pdf = norm.pdf(MC_dataNorm,0,1)
    return MC_data, MC_dataNorm, MC_dataNorm_pdf

def calculate_GMM_y_conditional_pdf(x,y,gmm):
    ''' Calculate the conditional pdf of y in the fitted Gaussian mixture model (GMM).
    
    Parameters
    -------
    x: scalar.
    y: array, shape (nMC,1).
    gmm : object. Fitted GMM.
    
    Returns
    -------
    yCondPDF : array, shape (nMC,1). Conditional pdf of y, f(y|x) in the given GMM.
    
    Notes
    -------
    # There are two approaches of calculating conditional pdf of y. 
    # Both approaches are coded here and have been tested working successfully.
    # However, method 1 is faster, so it is adopted in practice.
    - Equation number here is referred to paper: Hu, Z. and Mahadevan, S., 2019. 
    - Probability models for data-driven global sensitivity analysis. Reliability Engineering & System Safety, 187, pp.40-57.'''
    
    # Get attributes of the fitted GMM and y
    gmmWeights = gmm.weights_          # shape (n_components,)
    gmmMeans = gmm.means_              # shape (n_components, n_variables). n_variables = n_feature in sklearn.mixture.GaussianMixture reference.
    gmmCovariances = gmm.covariances_  # (n_components, n_variables, n_variables) if covariance_type = ‘full’ (by default).    
    gmmNComponents = gmm.n_components  # number of components
    nMC = len(y)                       # number of Monte Carlo samples

    # Method 1. use the relationship f(y|x) = f(x,y)/f(x) 
    # step 1. calculate f(x,y), joint pdf of (x,y) of the fitted GMM.
    # step 2. calculate f(x), marginal pdf of x of the fitted GMM.
    # step 3. calculate f(y|x), conditional pdf of y on x of the fitted GMM.

    # step 1. calculate f(x,y), joint pdf of (x,y) of the fitted GMM.
    multivariateData = np.concatenate((np.ones((nMC,1))*x,y), axis=1)  # combine x and y into a multi-variate data array. Shape (nMC, nVar+1).
    logProb = gmm.score_samples(multivariateData)                      # compute the log probability of multivariateData under the model.
    xyJointPDF = np.exp(logProb)                                       # get the joint probability of multivariateData of GMM.
    
    # step 2. calculate f(x), marginal pdf of x in the the fitted GMM.
    xMarginalPDFCpnt = [multivariate_normal.pdf(x, mean=gmmMeans[iComponent,:-1], cov=gmmCovariances[iComponent,:-1,:-1]) for iComponent in range(gmmNComponents)] # φ(x), shape(nComponent).
    xMarginalPDF = sum(xMarginalPDFCpnt*gmmWeights)
    
    # step 3. calculate f(y|x), conditional pdf of y on x in the fitted GMM.    
    yCondPDF = np.divide(xyJointPDF,xMarginalPDF)
    yCondPDF = yCondPDF.reshape(-1,1)

#     # Method 2. follow Eqs 32-35 of Hu, Z. and Mahadevan, S., 2019. 
#     # Step 1. calculate conditional weight of the GMM components given x (Eq 35).
#     # Step 2. calculate conditional mean and variance of y on x (Eqs 33-34).
#     # Step 3. calculate conditional pdf of y on x, f(y|x) (Eq 32).

#     # Step 1. calcualte conditional weights of the GMM components given x, λ(x) (Eq 35). Note λ(x)!=λ.
#     xMarginalPDFCpnt = [multivariate_normal.pdf(x, mean=gmmMeans[iComponent,:-1], cov=gmmCovariances[iComponent,:-1,:-1]) for iComponent in range(gmmNComponents)] # φ(x), shape(nComponent).
#     xMarginalPDFCpnt_weighted = xMarginalPDFCpnt*gmmWeights # λ*φ(x). shape(nComponent).  
#     print(np.shape(xMarginalPDFCpnt_weighted))
#     condGmmWeights = xMarginalPDFCpnt_weighted/sum(xMarginalPDFCpnt_weighted) # λ(x) = λ*φ(x)/sum(λ*φ(x)). shape(nComponent,).
#     condGmmWeights = np.reshape(condGmmWeights,(len(condGmmWeights),1)) # reshape to (nComponent,1) for dot multiplication.

#     # Loop steps 2 and 3 for each GMM component.
#     yCondPDFCpnt = np.zeros((nMC, gmmNComponents)) # f(y|x) for each GMM component.
#     for iCompoment in range(gmmNComponents):            
#         # Step 2. calculate conditional mean and variance of y on x (Eqs 33-34).
#         xyCov = gmmCovariances[iCompoment,-1,:-1]                                                   # covariance between x and y.
#         xxCov = gmmCovariances[iCompoment,:-1,:-1]                                                  # (co)variance of x.
#         xMean = gmmMeans[iCompoment,:-1]                                                            # mean of x.
#         yCondMean = gmmMeans[iCompoment,-1] + xyCov@np.linalg.inv(xxCov)@(x-xMean)                  # conditional mean of y, (Eq 33).
#         yCondVar = gmmCovariances[iCompoment,-1,-1] - xyCov@np.linalg.inv(xxCov)@xyCov.transpose()  # conditional variance of y, (Eq 34).

#         # Step 3. calculate conditional pdf of y on x, f(y|x) in each GMM component (Eq 32).
#         yCondPDFCpnt[:,iCompoment] = (multivariate_normal.pdf(y,mean=yCondMean, cov=yCondVar))      # (Eq 32)

#     # Finally. calculate mean f(y|x) over all GMM components via weighted sum λ(x)*f(y|x).
#     yCondPDF = yCondPDFCpnt@condGmmWeights                                                          # (Eq 32). shape(nMC,1)

    return yCondPDF

def GMCM_GSA(x,y,xNorm,yNorm,sensType,GSAIndex,nMC):
    """ Gaussian Mixture Copula-Based Estimator for first-order and total-effect sensitivity indices.
        reference: https://scikit-learn.org/0.16/modules/generated/sklearn.mixture.GMM.html
        reference: https://stackoverflow.com/questions/67656842/cumulative-distribution-function-cdf-in-scikit-learn
    
    Parameters
    -------
    x: scalar or array, shape (n_samples, n_variables). X values in normal space. 
    y: scalar or array. Y values in normal space. 
       - Recall that when fitting GMM, x (variable) and y (response) are combined to be the multivariates of GMM.
    sensType: str. Type of Sensitivity index calculation. Two options: first, total.
    GSAIndex: list. List of indices of x variable groups for sensitivity analysis. eg, [[0]], or [[0],[1],[2]], or [[0,1],[0,2],[1,2]]. 
    nMC: int. Number of Monte Carlo samples. 
    
    Returns
    -------
    sensIndex : array, shape (nGSAGroup,). Sensitivity index estimated using GMCM.
    
    Notes
    -------
    - Equation number here is referred to paper: Hu, Z. and Mahadevan, S., 2019. 
    - Probability models for data-driven global sensitivity analysis. Reliability Engineering & System Safety, 187, pp.40-57."""

    # Part 1. Standardize x and y samples.
    [nSample, nVar] = np.shape(x)
#     xNorm = np.zeros_like(x)     
#     for iVar in range(nVar):
#         xNorm[:,iVar] = standardize_data(x[:,iVar])         
#     yNorm = standardize_data(y)


    # Part 2. Calculate y variance and generate Monte Carlo y samples in normal space.
    varY = np.var(y)
    MC_y, MC_yNorm, MC_yNorm_pdf = generate_data_sample(y, nMC)
    
#     # Generate nMC y samples in the uniform manner from CDF [0,1].
#     # Note: Uniformly sample y CDF, not y, because the integration/sum is based on v=cdf(y).
#     MC_yCDF = np.linspace(0.0001,0.999,num=nMC)               # cdf of y in range of (0,1).
#     MC_yCDF = MC_yCDF.reshape(-1,1)                           # reshape MC_yCDF into (nMC,1).           

#     MC_yNorm = norm.ppf(MC_yCDF,0,1)                          # inverse of cdf in normal space, ie, F^(-1), percent point function.
#     MC_yNorm_pdf = norm.pdf(MC_yNorm,0,1)                     # pdf of MC_yNorm in normal space.

#     MC_y = np.zeros_like(MC_yCDF)                             # y in original space.
#     pctls = MC_yCDF[:,0]*100                                  # convert cdf to percentiles in range of (0,100).
#     MC_y[:,0] = np.percentile(y, pctls)                       # calculate y in original space based on the observed y(response) samples.
    

    # Part 3. Calculate sensitivity index.    
    if sensType == 'first':
        print('Calculating first-order sensitivity indices...')
    elif sensType == 'total':
        print('Calculating total-effect sensitivity indices...')        

    nGSAGroup = len(GSAIndex)           # total number of variable groups for sensitivity analysis
    sensIndex = np.zeros((nGSAGroup,))  # output array to store sensitivity index  
    sensIndex2 = np.zeros((nGSAGroup,)) # approach 2 output

    # Loop variable groups
    for iGSA in range(nGSAGroup):
        print('--- variable group %s ---'%(GSAIndex[iGSA]))

        # (1) Identify to-be-evaluated iGSA_xNorm
        if sensType == 'first':
            iGSA_xNorm = xNorm[:,GSAIndex[iGSA]]        

        elif sensType == 'total':               
            drop_cols = GSAIndex[iGSA] # drop columns first
            iGSA_xNorm = np.delete(xNorm, drop_cols, axis=1)

        # (2) Build the joint PDF of GMM (gmm_pdf) by fitting Gaussian density components to x and y in normal space.
        fitted_gmm = fit_GMM(iGSA_xNorm, yNorm)
        
        if not (fitted_gmm.converged_):
            print("ERROR: GMM fitting is not converged.")
            sys.exit(0)

        # (3) Generate two sets of nMC x samples based on the fitted GMM.
        # MC_xyNorm, MC_componentLabel = fitted_gmm.sample(nMC)               # Don't use the gmm built-in sample function because its random seed is fixed.
        MC_xyNorm1, MC_componentLabel1 = generate_GMM_sample(fitted_gmm, nMC) # return data = (x,y) and each data corresponding component label    .
        MC_xNorm1 = MC_xyNorm1[:,0:-1]                                        # keep only x.

        # (4) Calculate Var(E(y|x)): variance of the expectation of y conditioned on x (Eq 24). 
        # --- First loop of x samples to get E(y|x) (Eq 54).
        condEy1 = np.zeros((nMC,1))
        condVarY1 = np.zeros((nMC,1))
        
#         pbar = tqdm(total=nMC)
        for iMC in range(nMC):

            # Get the iMC^th x sample. 
            iMC_xNorm = MC_xNorm1[iMC,:]  

            # Given x, calculate conditioanl pdf of y, f(y|x). Apply to all MC_yNorm samples (y loop).
            MC_yNorm_gmmCondPDF = calculate_GMM_y_conditional_pdf(iMC_xNorm, MC_yNorm, fitted_gmm) 

            # Given x, calculate conditional expectation of y, E(y|x) (Eq 54). Apply to all MC_yNorm samples (y loop).
            iMC_condEy = sum(MC_y*(1/MC_yNorm_pdf)*MC_yNorm_gmmCondPDF)/float(nMC)   
            iMC_condEy1_sqaure = sum(MC_y*MC_y*(1/MC_yNorm_pdf)*MC_yNorm_gmmCondPDF)/float(nMC)

            # Save E(y|x).   
            condEy1[iMC] = iMC_condEy 
            condVarY1[iMC] = iMC_condEy1_sqaure - iMC_condEy*iMC_condEy

#             pbar.update(1)
#         pbar.close()                                  

        # --- Calculate Var(E(y|x)) (Eq 24).
        varCondEy = np.var(condEy1) 
        
        # --- Calculate E(Var(y|x)) (Eq 28).
        meanCondVarY = np.mean(condVarY1)

        # (6) Calulcate sensitivity index.   
        if sensType == 'first':
            iGSA_s = varCondEy/varY          # (Eq 3)  
            iGSA_s2 = 1 - meanCondVarY/varY  # (Eq 12)
            
        elif sensType == 'total':
            iGSA_s = 1-varCondEy/varY      # (Eq 4)   
            iGSA_s2 = meanCondVarY/varY    # (Eq 5)   

        sensIndex[iGSA] = iGSA_s           # save result.  
        sensIndex2[iGSA] = iGSA_s2
        
        print(iGSA_s,iGSA_s2)
    return sensIndex,sensIndex2

def plot_GMM_clusters(x,y,gmm,ofile,title):
    # reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
    # reference: https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/   
    
    data = np.concatenate((x, y), axis=1) # Combine x and y as multivariates of the Gaussian mixture model.

    # predictions from gmm
    labels = gmm.predict(data)
    frame = pd.DataFrame(data)
    frame['cluster'] = labels
    frame.columns = ['X', 'Y', 'cluster']

    fig, ax = plt.subplots(figsize=(9, 9*0.75))
    plt.title(title)

    scatter = ax.scatter(frame["X"], frame["Y"], c=frame["cluster"], s=2,cmap="jet")
    # Produce a legend for the ranking (colors). Even though there are many different
    # rankings, we only want to show 5 of them in the legend.
    legend1 = ax.legend(*scatter.legend_elements(),ncol=5,loc="upper left", title="Cluster")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
#     ax.set_ylim(-4,4)
    plt.savefig(ofile)
    plt.show()    
        
    return