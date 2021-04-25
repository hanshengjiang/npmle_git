#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:32:07 2021

@author: hanshengjiang
"""
from package_import import *
from simulation_lib import *
from cv_procedure_lib import *
from em_alg_lib import *
from regression_func_lib import *
import sys
from plotting_lib import *

'''
Ridgeline plots
'''

# ----------------------------
# python run_ridgeline_plot.py 1 homo
# ----------------------------


if __name__ == "__main__":
    # default
    if len(sys.argv) < 3:
        config = '2'
        error_type = 'homo' # error type can be hetero for config = 1,2,3
    # otherwise take argyments from command line
    else:
        #sys_argv[0] is the name of the .py file
        config = sys.argv[1]
        error_type = sys.argv[2]
        
# preset configurations      
if config == '1':
    #----------- configuration 1-----#
    b1 = (1,1)
    b2 = (4,-1)
    b3 = (-1,0.5)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[1,4],[1,-1]]
    alpha_true = [0.5,0.5]
    func = lin_func
    BL = -10
    BR = 10
elif config == '2':
    #----------- configuration 2-----#
    b1 = (0,1)
    b2 = (2,1)
    b3 = (0,1)
    pi1 = 0.3
    pi2 = 0.7
    B_true = [[0,2],[1,1]]
    alpha_true = [0.3,0.7]
    func = lin_func
    BL = -10
    BR = 10
elif config == '3':
    #----------- configuration 3-----#
    b1 = (3,-1)
    b2 = (1,1.5)
    b3 = (-1,0.5)
    pi1 = 0.3
    pi2 = 0.3
    B_true = [[3,1,-1],[-1,1.5,0.5]]
    alpha_true = [0.3,0.3,0.4]
    func = lin_func
    BL = -10
    BR = 10
elif config == '4':
    #----------- configuration 4-----# 
    b1 = (-4,1)
    b2 = (1, -1)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-4,1],[1,-1]]
    alpha_true = [0.5,0.5]
    func = poly_func
    BL = -10
    BR = 10
elif config == '5':
    #----------- configuration 5-----#

    b1 = (-0.5,-1)
    b2 = (-1.5,-1.5)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-0.5,-1.5],[-1,-1.5]]
    alpha_true = [0.5,0.5]
    func = exp_func
    BL = -10 #BL 
    BR = 10
elif config == '6':
    #----------- configuration 6-----#
    b1 = (-0.5,1)
    b2 = (-1.5,1.5)
    b3 = (0,0)
    pi1 = 0.5
    pi2 = 0.5
    B_true = [[-0.5,-1.5],[1,1.5]]
    alpha_true = [0.5,0.5]
    func = sin_func
    BL = -10
    BR = 10
elif config == '7':
    #----------- continuous case-----#
    meanb1 = [1,0.5]
    covb1 = np.array([[0.5,0.2],[0.2,0.3]])
    meanb2 = [2,3]
    covb2 = np.array([[0.5,0.2],[0.2,0.3]])
    pi1 = 1.0
    df_ = 3
    
    func = lin_func
    BL = -10
    BR = 10
    fname = 'continuous_' + str(meanb1[0]) + '_'+ str(meanb1[1])+'_'+ str(meanb2[0]) \
        +'_' +str(meanb2[1])+'_'+str(int(100*pi1)) +'percent'
    fname = fname.replace('.','dot')  
else:
    sys.exit("Wrong configuration number!")

if config != '7':
    fname = func.__name__[:-4] + str(b1[0]) + '_'+ str(b1[1])+'_'+ str(b2[0]) \
    +'_' +str(b2[1])+'_'+str(int(100*pi1)) +'percent'
    fname = fname.replace('.','dot')

if error_type == 'hetero':
    fname = 'hetero_'+fname
#------------------------------------------------------------------------

x_list_dense = np.arange(-1,3.1,0.1)

#----------------------True-------------------------------------------#

sigma = pd.read_csv('./../data/{}/sigma_true.csv'.format(fname), header = None).values.ravel()
    
if config != '7': 
    B1 = pd.read_csv('./../data/{}/B_true.csv'.format(fname), header = None).values
    alpha1 = pd.read_csv('./../data/{}/alpha_true.csv'.format(fname), header = None).values
   
    # set up range of y 
    min_ = -1
    max_ = 5
    for i in range(len(x_list_dense)):
        x = x_list_dense[i]
        min_ = min(min_, min([func([1,x],B1[:,i]) for i in range(len(B1[0]))]))
        max_ = max(max_, max([func([1,x],B1[:,i]) for i in range(len(B1[0]))]))
    min_ = float(int(min_))
    max_ = float(int(max_))
        
    df_true = density_ridgeline_plot(x_list_dense,sigma,\
                                        B1,alpha1,fname,min_,max_,func , approach = 'true')  
else:
    min_ = -2
    max_ = 8
    df_true = density_ridgeline_plot_continuous(x_list_dense,sigma,\
                        meanb1,covb1, meanb2,covb2,df_,pi1,fname, min_,max_,func, approach = 'true')
 
#----------------------NPMLE-sigma --------------------------------------#
B1 = pd.read_csv('./../data/{}/B_NPMLEsigma.csv'.format(fname), header = None).values
alpha1 = pd.read_csv('./../data/{}/alpha_NPMLEsigma.csv'.format(fname), header = None).values
sigma_NPMLEsigma = pd.read_csv('./../data/{}/sigma_NPMLEsigma.csv'.format(fname), header = None).values.ravel()

df_NPMLEsigma = density_ridgeline_plot(x_list_dense,sigma_NPMLEsigma,\
                                    B1,alpha1,fname, min_,max_,func, approach = 'NPMLEsigma')  



#----------------------NPMLE-CV --------------------------------------#
B2 = pd.read_csv('./../data/{}/B_NPMLE.csv'.format(fname), header = None).values
alpha2 = pd.read_csv('./../data/{}/alpha_NPMLE.csv'.format(fname), header = None).values
sigma_cv = pd.read_csv('./../data/{}/sigma_CV.csv'.format(fname), header = None).values.ravel()

df_NPMLE = density_ridgeline_plot(x_list_dense,sigma_cv,\
                                    B2,alpha2,fname, min_,max_,func, approach = 'NPMLE-CV')  


#-----------------------   EM-true  -------------------------------------#
if func.__name__ == 'lin_func' and config != '7':
    B4 = pd.read_csv('./../data/{}/B_EM.csv'.format(fname), header = None).values
    alpha4 = pd.read_csv('./../data/{}/alpha_EM.csv'.format(fname), header = None).values
    sigma_EM = pd.read_csv('./../data/{}/sigma_EM.csv'.format(fname), header = None).values.ravel()
    
    df_EM = density_ridgeline_plot(x_list_dense,sigma_EM,\
                                        B4,alpha4,fname, min_,max_,func, approach = 'EM-true')  

#-------------------------------------------------------------------#

# put ALL density functions in the same ridgeline plot
# NOTE: we will transpose the data frames




line_styles = ['-','-','-','-']
color_list = ['tab:gray', 'tab:green', 'tab:orange', 'tab:pink']
name_list = ['Truth',r'NPMLE-$\sigma$','NPMLE-CV', 'EM-true']

plot_config_list = [[True, False, False], [False, True, False], [False, False, True]]

if func.__name__ == 'lin_func':
    y = np.linspace(min_ -1, max_ + 1, 100)
    plt_index = [0,1,2,3]
else:
    y = np.linspace(min_ -3, max_ + 3, 100)
    plt_index = [0,1,2]

fig = plt.figure(figsize = (5*len(plt_index)-5,5))
axes = fig.subplots(nrows=1, ncols=len(plt_index)-1)

j_ = 0 
for ax in fig.axes:
    
    [plot_NPMLEsigma,plot_NPMLE,plot_EM] = plot_config_list[j_]
    j_ = j_ + 1
    
    # fig.add_subplot(1,len(plot_config_list)+1,i+1)
       
    curve_gap = 0.1
    for i in range(len(x_list_dense)):
        x_step = -1 + i * curve_gap
        
        curve_true = df_true.values[:,i]
        ax.plot(y, curve_true + x_step, color = color_list[0], linestyle = line_styles[0],\
                     zorder = len(x_list_dense)-i+1)
        if plot_NPMLEsigma == True:
            curve_NPMLEsigma = df_NPMLEsigma.values[:,i]
            ax.plot(y, curve_NPMLEsigma + x_step, color = color_list[1], \
                         zorder = len(x_list_dense)-i+1)
        if plot_NPMLE == True:
            curve_NPMLE = df_NPMLE.values[:,i]
            ax.plot(y, curve_NPMLE + x_step, color = color_list[2], \
                         zorder = len(x_list_dense)-i+1)
        if plot_EM == True:
            curve_EM = df_EM.values[:,i]
            ax.plot(y, curve_EM + x_step, color = color_list[3], \
                 zorder = len(x_list_dense)-i+1)
            
        ax.set_xlabel(r'$y$')
        
axes[0].set_ylabel(r'$x$')

# legend      
custom_lines = [
            Line2D([0], [0], color= color_list[0], linestyle = line_styles[0]),
          Line2D([0], [0], color= color_list[1], linestyle = line_styles[1]),
          Line2D([0], [0], color= color_list[2], linestyle = line_styles[2]),
          Line2D([0], [0], color= color_list[3], linestyle = line_styles[3])
    ]

axes[0].legend(np.array(custom_lines)[plt_index], np.array(name_list)[plt_index], \
                bbox_to_anchor=(0, -0.2), loc=2, ncol = len(plt_index),borderaxespad=0.)
fig.savefig("./../pics/ridgeline_{}_{}".format(fname, '_'.join(np.array(name_list)).replace('-$\sigma$', 'sigma')), \
                dpi = 300, bbox_inches='tight')





    
    
    
    
    
    


