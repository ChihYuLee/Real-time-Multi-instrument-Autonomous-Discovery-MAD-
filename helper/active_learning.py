import pylab as pb
import GPy
import numpy as np
import os 
from helper.utils import * 
import time 
from scipy.stats import multivariate_normal, entropy, norm
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")


def no_decision_making(LOG: dict, DATA: dict) -> tuple:
    """
    For random sampling
    :LOG: dict
    :DATA: dict
    return a tuple 
    """
    ####### sp ########
    measured_structure_idx = LOG['measured_s']
    all_idx = np.arange(len(DATA['XY'])) 
    candidate_s_idx = np.setdiff1d(all_idx, measured_structure_idx, assume_unique=True)
    #np.random.seed(DATA['RANDOM_SEED_S'])
    next_structure_idx = np.random.choice(candidate_s_idx)

    ###### fp ##########
    measured_functional_idx = LOG['measured_f']
    all_idx = np.arange(len(DATA['XY']))  
    candidate_f_idx = np.setdiff1d(all_idx, measured_functional_idx, assume_unique=True)
    #np.random.seed(DATA['RANDOM_SEED_F'])
    next_functional_idx = np.random.choice(candidate_f_idx)
    

    return [next_structure_idx, next_functional_idx]



def dicision_making(pred_mean: np.array, pred_std: np.array, LOG: dict, DATA: dict, 
    acq_p='max uncertainty', acq_f= 'max mean')-> tuple:
    """
    Decision making based on aquisition function 
    :pred_mean: (num of all comp, num phase comp+ num of fp)
    :pred_std: (num of all comp, num phase comp+ num of fp)
    :LOG: dict
    :DATA: dict
    :acq_p: acquisition function of pm 
    :acq_f: acquisition function of fm
    return next measure index [sp_next, fp_next]
    """
    #####  get predictions #####
    gpc_est = pred_mean[:, :DATA['NUM_COMP']]
    gpc_ent = pred_std[:, :DATA['NUM_COMP']]
    gpr_mean = pred_mean[:, DATA['NUM_COMP']:]
    gpr_std = pred_std[:, DATA['NUM_COMP']:]

    ##### acquisition function sp ######
    if acq_p =='ucb':
        kappa = 1.0  #balance 
        ucb = gpc_est + kappa * gpc_ent  
        structure_ucb_score = np.mean(ucb, axis=1)
        structure_idx = np.argsort(-structure_ucb_score)
    elif acq_p =='max uncertainty':
        #structure_mean_score = np.mean(gpc_ent, axis=1)
        structure_mean_score= np.sqrt(np.sum(gpc_ent, axis=1)) # square root (sum variance)
        structure_idx = np.argsort(-structure_mean_score)
    measured_structure_idx = LOG['measured_s']
    structure_idx = np.setdiff1d(structure_idx, measured_structure_idx, assume_unique=True) #not re-order 
    next_structure_idx = structure_idx[0]

    ##### acquisition function fp #########
    mean1= gpr_mean
    cov1=  gpr_std
    if acq_f  =='ei':
        #max_mean_y = np.max(mean1)    # all maximum 
        observed= DATA['known_f']
        max_mean_y= np.max(observed)# oberved maximum 
        z = (mean1 - max_mean_y) / cov1
        exp_imp = (mean1 - max_mean_y) * norm.cdf(z) + cov1 * norm.pdf(z)
        function_i= np.argsort(-exp_imp.flatten())
        measured_function_idx= LOG['measured_f']
        function_idx= np.setdiff1d(function_i, measured_function_idx, assume_unique=True)
        next_function_idx= function_idx[0]
    elif acq_f =='max mean':
        function_i= np.argsort(-mean1.flatten())
        measured_function_idx= LOG['measured_f']
        function_idx= np.setdiff1d(function_i, measured_function_idx, assume_unique=True)
        next_function_idx= function_idx[0]
    return [next_structure_idx, next_function_idx]

def update_data(FOLDER_PATH: str, i: int, pred_mean: np.array, pred_std: np.array, gp: dict, 
                 decisions: list, measured_f: list, measured_s:list,
                 xy: np.array,  sp_input: np.array, basis_matrix: np.array,
                 DATA: dict, LOG: dict, live_plot=True)->None:
    """
    Organize central data repository

    :FOLDER_PATH: the path to save
    :i: number of iteration 
    :pred_mean: (num of all comp, num phase comp+ num of fp)
    :pred_std: (num of all comp, num phase comp+ num of fp)
    :gp: gp model 
    :decisions: [next_sp, next_sp]
    :measured_f: list of measured index 
    :measured_s: list of meausred index 
    :xy: all comp coordinates 
    :sp_input: abundance input (rsw data)  !!different from sp_output (pred)
    :basis_matrix: the basis components 
    :DATA: dictionary 
    :LOG: dictionary
    return None 
    """
    ### LOG ####
    LOG['measured_s']= np.append(LOG['measured_s'], decisions[0])
    LOG['measured_f']= np.append(LOG['measured_f'], decisions[1])

    ### DATA ####
    DATA['known_s']= DATA['TRUE_S'][LOG['measured_s'],:]
    DATA['known_f']= DATA['TRUE_F'][LOG['measured_f'],:]

    #### PLOT ####
    gpc_est = pred_mean[:, :DATA['NUM_COMP']]
    gpc_ent = pred_std[:, :DATA['NUM_COMP']]
    gpr_mean = pred_mean[:, DATA['NUM_COMP']:]
    gpr_std = pred_std[:, DATA['NUM_COMP']:]

    xy= cart2tern(xy.copy())

    # calculate all 
    y_true= DATA['TRUE_F']
    y_pred= gpr_mean
    rmse_1 = np.sqrt(np.mean((y_true - y_pred) ** 2))
    percent_err = np.mean(np.abs((y_pred - y_true) / y_true))*100
    global_max= np.max(y_true)
    global_max_ohm= 10**global_max
    #temp within all
    #temp_global_max= np.argmax(y_pred)
    #temp within the measured 
    temp_global_max= np.max(y_pred[measured_f])
    temp_global_max_ohm= 10** temp_global_max
    risk= np.linalg.norm(global_max_ohm - temp_global_max_ohm)

    y_true_s= DATA['TRUE_S']
    y_pred_s= gpc_est
    reconstructed_xrd= np.dot(y_pred_s, basis_matrix)
    similarity = cosine_similarity(reconstructed_xrd, DATA['RAW_S'])
    similarity_score= similarity[0][0]
    dtw= calculate_dtw(reconstructed_xrd, DATA['RAW_S'])

    # calculte only measured 
    rmse_fp_measured= np.sqrt(np.mean((y_true[measured_f,:]- y_pred[measured_f,:]) ** 2))
    percent_err_measured = np.mean(np.abs((y_pred[measured_f,:] - y_true[measured_f,:]) / y_true[measured_f,:]))*100
    similarity_measured = cosine_similarity(reconstructed_xrd[measured_s,:], DATA['RAW_S'][measured_s,:])
    dtw_measured= calculate_dtw(reconstructed_xrd[measured_s,:], DATA['RAW_S'][measured_s,:])
    

    # others 
    sp= y_pred_s[LOG['measured_s'],:]
    f_to_plot='correlation'
    corr= gp_model(gp, f=f_to_plot)

    LOG['log'][i]={'rmse_fp': rmse_1, 'risk': risk, 'similarity': similarity,'dtw': dtw, 
                    'rmse_fp_measured':  rmse_fp_measured, 'percent_error': percent_err, 
                    'percent_error_measured' :percent_err_measured , 'similarity_measured': similarity_measured, 'dtw_measured':dtw_measured, 
                    'sp_input': sp_input, 'sp_output': sp, 
                    'basis_matrix': basis_matrix, 'gp': gp, 'latent_corr': corr,
                    'gpc_mean': gpc_est, 'gpc_std': gpc_ent,
                    'gpr_mean':gpr_mean, 'gpr_std':gpr_std}


    # for live run (data focus)
    fig, axes = plt.subplots(2, 4, dpi=300, facecolor="white", figsize=((4*2.5, 2*2.3)), constrained_layout=False)
    fig.suptitle('Iteration {} (No. samples={}, {}+{})'.format(i+1,LOG['measured_s'].shape[0],LOG['measured_s'].shape[0]-DATA['INIT'], DATA['INIT']), fontsize=15,fontweight="bold")

    ## Fig 1 phase mapping 
    draw_piechart(axes[0][0], xy[LOG['measured_s'],:], sp.shape[1], sp, circle=True)
    axes[0][0].set_title('Phase Mapping', fontsize=12, pad=15, c='darkred')

    ## Fig 2 xrd reconstruction
    scale = 1.8  # 30% wider
    pos = axes[0][1].get_position()
    extra = (scale - 1) * pos.width / 2
    new_pos = [pos.x0 - extra, pos.y0, pos.width * scale, pos.height]
    axes[0][1].set_position(new_pos)
    axes[0][1].plot(DATA['XRD_theta'],reconstructed_xrd[LOG['measured_s'][-1],:], c='darkred', label='reconstructed')
    axes[0][1].plot(DATA['XRD_theta'],DATA['RAW_S'][LOG['measured_s'][-1],:], c='black', label='raw')
    axes[0][1].set_title('XRD reconstruction of {}'.format(LOG['measured_s'][-1]), fontsize=12, pad=15, c='darkred')
    axes[0][1].set_xlabel(r"$2\theta$ (°)", fontsize=10)
    axes[0][1].set_ylabel("Intensity (a.u.)", fontsize=10)
    axes[0][1].set_ylim((0,3000))
    axes[0][1].set_yticklabels([])
    l=axes[0][1].legend(fontsize=7, loc='upper left')
    l.get_frame().set_facecolor('none')
    l.get_frame().set_edgecolor('white')

    ## Fig 3 basis component 
    parent_ax = axes[1][0]
    #parent_ax.yaxis.set_visible(False)
    parent_ax.set_yticklabels([])
    parent_ax.set_yticks([])
    parent_ax.tick_params(axis='y', which='both', left=False)
    parent_ax.xaxis.set_visible(False)
    parent_ax.set_title('Basis Component', fontsize=12, pad=12, c='darkred')
    n = DATA['NUM_COMP']
    inset_axes_list = []

    # Get the position of parent axes in figure coordinates
    pos = parent_ax.get_position()
    fig = parent_ax.figure
    parent_ax.remove()

    shrink = 0.75
    y_shift = 0.01     # ↓ move down (figure fraction)
    new_height = pos.height * shrink
    new_y0 = pos.y0 + (pos.height - new_height)/2 - y_shift

    inset_axes_list = []
    for ii in range(n):
        h = new_height / n
        y0 = new_y0 + (n - 1 - ii) * h
        ax = fig.add_axes([pos.x0, y0, pos.width, h])
        inset_axes_list.append(ax)
    # Plot each series
    cmap= cm.get_cmap("viridis", n) 
    colors = [cmap(ii) for ii in range(n)]
    for ii, ax in enumerate(inset_axes_list):
        ax.plot(DATA['XRD_theta'], basis_matrix[ii,:], c=colors[ii],label=f"C{ii}")
        ax.set_xlim(np.min(DATA['XRD_theta']), np.max(DATA['XRD_theta']))
        if ii!= int(len(inset_axes_list)-1):
            ax.set_xticklabels([])


        #ax.xaxis.set_visible(False)
        #if ii== len(inset_axes_list):
        #    ax.set_xlabel(r"$2\theta$ (°)", fontsize=10)
        #else:
        #    ax.set_xticklabels([])
    
        
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.tick_params(axis='y', which='both', left=False)
        ax.text(
        -0.08, 0.5,           # x slightly outside axis
        f"C{ii}",             # your label
        transform=ax.transAxes,
        fontsize=9,
        va='center',
        ha='right'
    )
        #l=ax.legend(fontsize=4)
        #l.get_frame().set_edgecolor('black')
    ax.set_title('Basis Component', fontsize=12, pad=82, c='darkred')
    ax.set_xlabel(r"$2\theta$ (°)", fontsize=10)

    ## Fig 4 xrd reconstruction score (Cosine)
    axes[1][1].set_title('Reconstruction vs.Raw (all)', fontsize=12, pad=10, c='darkred')
    axes[1][1].set_ylim((0,1))
    axes[1][1].set_xlabel('Iteration')
    axes[1][1].set_xlim((1, DATA['LOOP']))
    axes[1][1].set_ylabel('CSS',c='#39568CFF', fontsize=10)
    axes[1][1].tick_params(axis='y', labelsize=8)
    axes[1][1].tick_params(axis='x', labelsize=6)
    i_list = list(LOG['log'].keys())
    sm_list = [LOG['log'][j]['similarity'][0][0] for j in i_list]
    i_list_plus1= [int(i+1) for i in i_list]
    axes[1][1].plot(i_list_plus1, sm_list, marker='o', c='#39568CFF', label='Cosine similarity')
    axes[1][1].xaxis.set_major_locator(MaxNLocator(integer=True))
    #step = 5  # show every 2nd tick
    #xticks = i_list_plus1[::step]
    #axes[1][1].set_xticks(xticks)
    #axes[1][1].set_xticklabels([str(x) for x in xticks])
    axes[1][1].tick_params(axis='y', labelcolor='#39568CFF')
    ax2 = axes[1][1].twinx()
    ax2.set_ylabel("DTW (a.u.)", color='#20A387FF', fontsize=10)
    dtw_list = [LOG['log'][j]['dtw'] for j in i_list] 
    ax2.plot(i_list_plus1, dtw_list, marker='o', c='#20A387FF', label='DTW distance')
    #ax2.set_ylim((2000,10000))
    ax2.tick_params(axis='y', labelcolor='#20A387FF', labelsize=8)
    lines_1, labels_1 = axes[1][1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    #l=axes[1][1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=7)
    #l.get_frame().set_edgecolor('white')


    ## Fig 5 functional property prediction 
    axes[0][2].set_title('Resistance Optimization', fontsize=12, pad=15, c='navy')
    td = Td(["Te", "Sb", "Mn"], ax=axes[0][2])
    sc= td.scatter(vector=xy[LOG['measured_f'],:], marker="o", c=DATA['known_f'],  edgecolors='black', vmin=0, vmax=7 ,s=20)
    current_p= td.scatter(vector=xy[LOG['measured_f'][-1],:].reshape(1, -1), facecolors='none', edgecolors='red', linewidths=1.5, s=150)
    cbar = plt.colorbar(sc, ax=axes[0][2], fraction=0.03, pad=0.1, location='left')
    cbar.set_label("Sheet resistance $\log_{10}(\Omega)$", fontsize=8)

    ## Fig 6 Correlation to FP 
    
    #im= axes[0][3].imshow(corr, vmin=-1, vmax=1, interpolation='nearest', origin='lower')
    #axes[0][3].set_title('Coregionalization Matrix', fontsize=10, pad=20)
    #cbar= plt.colorbar(im)
    #cbar.ax.tick_params(labelsize=6)
    #cbar.set_label('Correlation', fontsize=8)
    all_to_fp= corr[:-1,-1]
    labels = [f"C{i}" for i in range(all_to_fp.shape[0])]
    bars = axes[0][3].bar(labels, all_to_fp, color='skyblue', edgecolor='black')
    for bar, val in zip(bars, all_to_fp):
        if val < 0:
            bar.set_color("darkred")  # red for negative
        else:
            bar.set_color("black") # blue for positive 
    axes[0][3].set_title('Correlation to FP', fontsize=12, pad=15, c='navy')
    axes[0][3].set_ylim(-1,1)
    axes[0][3].set_ylabel('Correlation coefficient',fontsize=8 )
    axes[0][3].tick_params(axis='y', labelsize=8)

    ## Fig 7 True and Pred (all) 
    axes[1][2].set_title('True vs. Pred (all)', pad=12, fontsize=12,c='navy')
    axes[1][2].set_xlabel('Iteration')
    #axes[1][2].set_ylabel('RMSE ($\log_{10}(\Omega)$)',c='#39568CFF', fontsize=10)
    axes[1][2].set_ylabel('Relative Error (%)',c='#39568CFF', fontsize=10)
    axes[1][2].tick_params(axis='y', labelsize=8)
    axes[1][2].tick_params(axis='y', labelcolor='#39568CFF')
    axes[1][2].tick_params(axis='x', labelsize=6)
    axes[1][2].set_ylim((0,100))
    axes[1][2].set_xlim((1,DATA['LOOP']))
    i_list = list(LOG['log'].keys())
    risk_list = [LOG['log'][j]['risk'] for j in i_list]
    #rmse_list = [LOG['log'][j]['rmse_fp'] for j in i_list]
    pe_list=  [LOG['log'][j]['percent_error'] for j in i_list]
    i_list_plus1= [int(i+1) for i in i_list]
    #axes[1][2].plot(i_list_plus1, rmse_list, marker='o', c='#39568CFF')
    axes[1][2].plot(i_list_plus1, pe_list, marker='o', c='#39568CFF')
    #step = 5  # show every 2nd tick
    #xticks = i_list_plus1[::step]
    #axes[1][2].set_xticks(xticks)
    #axes[1][2].set_xticklabels([str(x) for x in xticks])
    ax2 = axes[1][2].twinx()
    ax2.set_ylabel("MR ($\Omega$) ", color='#20A387FF', fontsize=10)
    ax2.plot(i_list_plus1, risk_list, marker='o', c='#20A387FF')
    #ax2.set_ylim((0,3))
    ax2.tick_params(axis='y', labelcolor='#20A387FF', labelsize=8)
    #ax2.set_yscale('log')
    axes[1][2].xaxis.set_major_locator(MaxNLocator(integer=True))

    ## Fig 8 True and Pred (measured) 
    axes[1][3].set_title('True vs. Pred (measured)', pad=12, fontsize=12,c='navy')
    axes[1][3].set_xlabel('Iteration')
    axes[1][3].set_ylabel('Relative Error (%)',c='#39568CFF', fontsize=10)
    axes[1][3].set_ylim((0,100))
    axes[1][3].set_xlim((1,DATA['LOOP']))
    axes[1][3].tick_params(axis='y', labelsize=8)
    axes[1][3].tick_params(axis='y', labelcolor='#39568CFF')
    axes[1][3].tick_params(axis='x', labelsize=6)
    i_list = list(LOG['log'].keys())
    #rmse_list = [LOG['log'][j]['rmse_fp_measured'] for j in i_list]
    pe_measured_list=  [LOG['log'][j]['percent_error_measured'] for j in i_list]
    i_list_plus1= [int(i+1) for i in i_list]
    #axes[1][3].plot(i_list_plus1,rmse_list, marker='o', c='#39568CFF')
    axes[1][3].plot(i_list_plus1, pe_measured_list, marker='o', c='#39568CFF')
    axes[1][3].xaxis.set_major_locator(MaxNLocator(integer=True))
    #step = 5  # show every 2nd tick
    #xticks = i_list_plus1[::step]
    #axes[1][3].set_xticks(xticks)
    #axes[1][3].set_xticklabels([str(x) for x in xticks])
    
    plt.tight_layout()
    plt.savefig(FOLDER_PATH+'{}_live.png'.format(i), dpi=300, bbox_inches='tight')#,transparent=True)
    plt.show()

    ## Fig 1 for video
    fig1, axes1 = plt.subplots(1, 1, dpi=300, facecolor="white", figsize=((2.3, 2.3)), constrained_layout=False)
    draw_piechart(axes1, xy[LOG['measured_s'],:], sp.shape[1], sp, circle=True)
    axes1.set_title('Phase Mapping', fontsize=13,fontweight='bold', pad=10, c='black')
    plt.tight_layout()
    plt.savefig(FOLDER_PATH+'{}_pm.png'.format(i), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    ## Fig 5 for video 
    ####################
    fig2, axes2 = plt.subplots(1, 1, dpi=300, facecolor="white", figsize=((2.3, 2.3)), constrained_layout=False)
    axes2.set_title('Resistance Optimization', fontsize=13, fontweight='bold', pad=10, c='black')
    td = Td(["Te", "Sb", "Mn"], ax=axes2)
    sc= td.scatter(vector=xy[LOG['measured_f'],:], marker="o", c=DATA['known_f'],  edgecolors='black', vmin=0, vmax=7 ,s=20)
    current_p= td.scatter(vector=xy[LOG['measured_f'][-1],:].reshape(1, -1), facecolors='none', edgecolors='red', linewidths=1.5, s=150)
    cbar = plt.colorbar(sc, ax=axes2, fraction=0.03, pad=0.1)
    cbar.set_label("Sheet resistance $\log_{10}(\Omega)$", fontsize=8)
    plt.tight_layout()
    plt.savefig(FOLDER_PATH+'{}_fm.png'.format(i), dpi=300, bbox_inches='tight')
    plt.close(fig2)

