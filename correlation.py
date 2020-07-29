"""
cramer's v is symmetric nominal association.
"""
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    #print(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    #print(chi2)
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


"""
source:
https://github.com/shakedzy/dython/blob/master/dython/nominal.py
https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
"""
import math
import numpy as np
from collections import Counter
def conditional_entropy(x, y):
    _, yv, yc = np.unique(y, return_counts=True, return_inverse=True)
    xy = np.dstack([x, y])
    
    xy, xyv, xyc = np.unique(xy, return_counts=True, return_inverse=True, axis=1)
    
    entropy = -(np.log(xyc[xyv])-np.log(yc[yv])).sum()/len(x)#np.log(yc[yv]/xyc[xyv]).sum()/len(x)
    return entropy


"""
theil's u is asymmetric nominal association.
"""
def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    _, xv, xc = np.unique(x, return_counts=True, return_inverse=True)
    #print(xc, xv)
    #p_x = xc/len(x)
    
    #s_x = ss.entropy(p_x)
    ent = -(np.log(xc[xv])-np.log(len(x))).sum()/len(x)
    #assert np.isclose(mv, s_x) == True
    #assert s_x != 0
    
    res = (ent - s_xy) / ent if ent else 1
    #print(res)
    return res
    #return (s_x - s_xy) / s_x if s_x else 1
    
    
    
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta