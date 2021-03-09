#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""
from package_import import *
from alg1_lib import *

# read data
df = pd.read_csv('./../data/co2-emissions-vs-gdp.csv')
df = df[df['Code']!='TTO']
df = df[df['Code']!='QAT']
df = df[df['Entity']!='Taiwan']
df = df[df['Entity']!='Hong Kong']
dfs = df.dropna()
dataf = dfs.iloc[:,[5,6]].values.astype(np.float)
# 5 population in billions; 8: CO2 in millions ; GDP percapita in 1000USD 
region_list = dfs.iloc[:,1].values
# dataf[:,1:] = dataf[:,1:].astype(np.float)
n = np.shape(dataf)[0]
ones = np.ones((n,1))
X = np.concatenate((ones, np.reshape(dataf[:,0],(n,1))/10000 ), axis = 1) #GDP per capita (1000 USD)
y = np.reshape(dataf[:,1]/10,(n,1)) #CO2 per capita (10 ton)

iter = 200
threprob = 1e-2

#Use Algorithm 1
sigma = 0.31 # chosen by cross-validation procedure
np.random.seed(26)
f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma)
print("number of components", len(alpha))
##########IMPORTANT subproblem initializes with beta = 0

#plot
print("final neg log likelihood is ", L_final)
print("number of components is", len(alpha))
print("only components with probability at least ", threprob, " are shown below:")

fig1, ax = plt.subplots(figsize = (8,6))
plt.scatter(X[:,1],y,color = 'blue',marker = 'o', facecolors = 'None',linewidths = 0.5);
#label = 'Noisy data',

# for i, txt in enumerate(region_list):
#     if X[i,1] > 1:
#         ax.annotate(txt, (X[i,1],y[i])) # replace marker by text

nation_list = [ 'USA','CAN','KAZ','DNK','GBR','BHR','SGP','NOR']
# nation_list = region_list
corr = -0.1
for i, txt in enumerate(region_list):
    if txt in nation_list and i%1 == 0:
        ax.annotate(txt, (X[i,1] + corr, y[i] + corr),size = 12) # replace marker by text
    
t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-2)
i = 0
index_sorted = np.argsort(-np.reshape(alpha,(len(alpha),)))
for i in index_sorted:
    b = B[:,i]
    if alpha[i] >threprob:
        plt.plot(t,b[0]+b[1]*t, color = str((1-alpha[i][0])/100),linewidth = alpha[i][0]*8 ,\
                 label = '$y = %.4f + %.4f x$ with probability $%.2f$' %(b[0], b[1], alpha[i]) )
        print("coefficients", b, "with probability", alpha[i])
# custom_lines = [Line2D([], [], color='gray', marker='o',markerfacecolor = 'None', linestyle='None'),
#                 Line2D([0], [0], color='black')
#                 ,]
# ax.legend(custom_lines, ['Noisy data'
#                           , 'NPMLE component'
#                          ],loc=9);
ax = plt.gca()
ax.set_xlabel(r'$x$ ($\rm{GDP}$)')
ax.set_ylabel(r'$y$: ($\rm{CO_2}$)')
lgd = ax.legend(loc=9, bbox_to_anchor=(1.45, 1),borderaxespad=0.) 
plt.savefig('./../pics/co2_gdp.png', dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')