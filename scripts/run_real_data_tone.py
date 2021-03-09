#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:12:07 2021

"""
from package_import import *
from alg1_lib import *

#read tonedata into file
with open('./../data/tonedata.csv', newline='') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[1:] # remove row containing column name
dataf = data.astype(np.float)
n = np.shape(dataf)[0]
ones = np.ones((n,1))
X = np.concatenate((ones, np.reshape(dataf[:,0],(n,1))), axis = 1)
y = np.reshape(dataf[:,1],(n,1))


iter = 200
threprob = 0.02

#Use Frank-Wofle with an estimated sigma
sigma = 0.05
f, B, alpha, L_rec, L_final = NPMLE_FW(X,y,iter,sigma)


#beta_ols = np.reshape(np.dot(np.matmul(linalg.inv(np.matmul(X.T,X)),X.T),y),(p,1))

#plot
print("final neg log likelihood is ", L_final)
print("number of components is", len(alpha))
print("only components with probability at least ", threprob, " are shown below:")

fig1 = plt.figure(figsize = (8,8))
# label = 'Original data',
plt.scatter(X[:,1],y,color = 'blue',marker = 'o',linewidths=0.5, facecolors = 'None');
t = np.arange(np.amin(X[:,1])-0.5,np.amax(X[:,1])+0.5,1e-6)
#plt.plot(t,b1[0]+b1[1]*t,'r',t,b2[0]+b2[1]*t,'red')
i = 0

index_sorted = np.argsort(-np.reshape(alpha,(len(alpha),)))
for i in index_sorted:
    b = B[:,i]
    if alpha[i] >threprob:
        plt.plot(t,b[0]+b[1]*t, color = str((1-alpha[i][0])/100),linewidth = alpha[i][0]*8 ,\
                 label = r'$y = %.4f + %.4f x$ with probability $%.2f$' %(b[0], b[1], alpha[i]))
        print("coefficients", b, "with probability", alpha[i])
#plt.plot(t,beta_ols[0]+beta_ols[1]*t,'green')  
#plt.legend(custom_lines, ['Noisy data'#,'True mixture'# 
                         # , 'NPMLE component'#, 'OLS'#
                         #],loc=2);
ax = plt.gca()
lgd = ax.legend(loc=9, bbox_to_anchor=(1.45, 1),borderaxespad=0.) 
ax.set_xlabel(r'$x$: stretching ratio')
ax.set_ylabel(r'$y$: djusted ratio')
plt.savefig('./../pics/tone.png', dpi = 300, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show();

