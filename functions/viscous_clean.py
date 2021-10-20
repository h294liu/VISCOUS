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
from scipy.stats import norm, gaussian_kde, multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def define_GSA_variable_index(nVar):
    """ Create variable indices for which the variance-based sensitivity indices are estimated.
    GSA: global sensitivity analysis.
    
    Parameters
    -------
    nVar: int. Total number of input variables (eg, parameters).
    
    Returns
    -------
    GSAIndex : list. List of indices of x variable groups to be evaluated. eg, [[0]], or [[0],[1],[2]], or [[0,1],[0,2],[1,2]]. 
    
    Notes
    -------
    For now only the first-order sensitivity is calculated, thus [[1],[2],[3]].
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
    data: array, shape (n, 1) or (n,). Data array to be standardized.
    
    Returns
    -------
    dataNorm: array, shape is the sames as data. Standardized data.
    
    Notes
    -------
    A faster way of calculating CDF. 
    Modify source code of kde.integrate_box_1d, and make it process array, not element-by-element. BUT this requires a large memory. 
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

def generate_data_sample(data, n_samples):
    """ Sample random samples from input data using the kernel sensity function.
     
    Parameters
    -------
    data: array, shape (n, 1) or (n,). Data array in the original space (not standard Gaussian/normal space).
    n_samples : int. Number of samples to generate.
    
    Returns
    -------
    sample : array, shape (n_samples, 1). Randomly generated sample.
    sampleNorm: array, shape (n_samples, 1). Sample values in standrard normal space.
    sampleNorm_pdf: array, shape (n_samples, 1). PDF of sample in standrard normal space. """    

    data = data.reshape((1,len(data)))
    kde = gaussian_kde(data)
    sample = kde.resample(n_samples)                 # sample

    cdf_function = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    sampleCDF = cdf_function(sample)                 # calcualte cdf for sample
    sampleNorm = norm.ppf(sampleCDF, loc=0, scale=1) # calcualte inverse of sample cdf in the normal space

    sample = sample.T                                # reshape into (n_sample,1)
    sampleNorm = sampleNorm.T                        # reshape into (n_sample,1)
    sampleNorm_pdf = norm.pdf(sampleNorm,0,1)        # pdf of sampleNorm in the normal space
    
    return sample, sampleNorm, sampleNorm_pdf

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

def calculate_GMM_y_conditional_pdf(x,y,gmm):
    ''' Calculate the conditional pdf of y in the fitted Gaussian mixture model (GMM).
    
    Parameters
    -------
    x: scalar. X values in normal space. 
    y: array, shape (nMC,1). Y values in normal space. 
    gmm : object. Fitted GMM.
    
    Returns
    -------
    yCondPDF : array, shape (nMC,1). Conditional pdf of y, f(y|x) in the given GMM.
    
    Notes
    -------
    # There are two approaches of calculating conditional pdf of y. 
    # Both approaches are coded here and have been tested working successfully.
    # However, our presented method is faster, so it is adopted here.
    - The second method is based on Eqs 32-35 of Hu, Z. and Mahadevan, S., 2019. 
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
    
    # step 2. calculate f(x), marginal pdf of x in the the fitted GMM. shape(nComponent).
    xMarginalPDFCpnt = [multivariate_normal.pdf(x, mean=gmmMeans[iComponent,:-1], 
                                                cov=gmmCovariances[iComponent,:-1,:-1]) for iComponent in range(gmmNComponents)] 
    xMarginalPDF = sum(xMarginalPDFCpnt*gmmWeights)
    
    # step 3. calculate f(y|x), conditional pdf of y on x in the fitted GMM.    
    yCondPDF = np.divide(xyJointPDF,xMarginalPDF)
    yCondPDF = yCondPDF.reshape(-1,1)

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
    GSAIndex: list. List of indices of x variable groups to be evaluated. eg, [[0]], or [[0],[1],[2]], or [[0,1],[0,2],[1,2]]. 
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

    # Part 2. Calculate y variance and generate Monte Carlo y samples based on y.
    varY = np.var(y)
    MC_y, MC_yNorm, MC_yNorm_pdf = generate_data_sample(y, nMC)   

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
        # --- Loop x samples to get E(y|x) (Eq 54).
        condEy1 = np.zeros((nMC,1))
        condVarY1 = np.zeros((nMC,1))
        
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

        # --- Calculate Var(E(y|x)) (Eq 24).
        varCondEy = np.var(condEy1) 
        
        # --- Calculate E(Var(y|x)) (Eq 28).
        meanCondVarY = np.mean(condVarY1)

        # (6) Calulcate sensitivity index using two approaches.   
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
    """ Gaussian Mixture Copula-Based Estimator for first-order and total-effect sensitivity indices.
        reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#sphx-glr-gallery-lines-bars-and-markers-scatter-with-legend-py
        reference: https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/   
    
    Parameters
    -------
    x: scalar or array, shape (n, 1) or (n,). X values in normal space. 
    y: scalar or array, shape (n, 1) or (n,). Y values in normal space. 
    gmm : object. Fitted GMM.
    ofile: path. Path of output figure.
    title: str. Title of the output figure."""     
    
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
    plt.savefig(ofile)
    plt.show()    
        
    return