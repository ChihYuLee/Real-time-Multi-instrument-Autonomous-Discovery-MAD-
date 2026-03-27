import pylab as pb
import GPy
import numpy as np
import os 
from helper.utils import * 
from sklearn.preprocessing import StandardScaler
import time 



def coregionalization_scale(xy_pred: np.array, measured_f: np.array, measured_s: np.array, fp: np.array, sp: np.array, 
                            matrix= 'LCM', W= 1, scale=True, clip_positive=True, gp_verbose=True)-> tuple:
    """
    v2.0: scaling and just matern32
    Linear coregionalization matrix, regression input and output 
    Take phase abundances and functional property as input 
    :xy: 2d coordinates 
    :measured_f: index 
    :measured_s: index 
    :fp: (number of sample, 1) -> resistance 
    :sp: (number of sample, number of component) -> abundance
    :W: how complex the pattern of correlation
    :scale: whether to scale data or not 
    :clip_positive: to avoid negative output prediction, to fit NMF framework 
    return mean, std, gp_dict
    """ 
    start_time = time.time()
    
    if scale:
        scaler_fp = StandardScaler()
        scaler_sp = StandardScaler()
        fp_scaled = scaler_fp.fit_transform(fp)
        sp_scaled = scaler_sp.fit_transform(sp)
        input_xy= [ xy_pred[measured_s].copy() for _ in range(sp.shape[1])]+[xy_pred[measured_f]]
        output_z= [sp_scaled[:, i][:,None] for i in range(sp.shape[1])]+ [fp_scaled]
        output_mean=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        output_std=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        
        if matrix=='LCM':
            K = GPy.kern.Matern32(2)
            K.lengthscale.constrain_bounded(0.01, 2.0)
            K.variance.constrain_bounded(0.01, 2.0)
            lcm = GPy.util.multioutput.LCM(input_dim=2,num_outputs=sp.shape[1]+fp.shape[1],kernels_list=[K], W_rank=W)
            m = GPy.models.GPCoregionalizedRegression(input_xy, output_z,kernel=lcm)
            m['.*ICM.*var'].unconstrain()
            m['.*ICM.*W'].unconstrain()
            m.optimize(max_iters=5000)
        elif matrix =='ICM':
            K = GPy.kern.Matern32(2)
            K.lengthscale.constrain_bounded(0.01, 2.0)
            K.variance.constrain_bounded(0.01, 2.0)
            icm = GPy.util.multioutput.ICM(input_dim=2, num_outputs=sp.shape[1]+fp.shape[1], kernel=K, W_rank=W)
            m = GPy.models.GPCoregionalizedRegression(input_xy, output_z, kernel=icm)
            m['.*ICM.*var'].unconstrain()
            m['.*ICM.*W'].unconstrain()
            m.optimize(max_iters=5000)
        
        model_dict = m.to_dict(save_data=True)
        if gp_verbose:
            print(m)

        for i in range(sp.shape[1]+fp.shape[1]):
            newX = np.hstack([xy_pred, np.ones((xy_pred.shape[0], 1)) * i])  # shape (177, 2)
            noise_dict = {'output_index': newX[:, 2:].astype(int)}  # shape (177, 1)
            m_, v_ = m.predict(newX, Y_metadata=noise_dict)  # m_: shape (177, 1)
            output_mean[:,i]= m_.flatten()
            output_std[:,i]= v_.flatten()
        
        sp_dim = sp.shape[1]
        output_mean[:, :sp_dim] = scaler_sp.inverse_transform(output_mean[:, :sp_dim]) #fix negative prediction although very samll 
        output_std[:, :sp_dim] *= scaler_sp.scale_
        output_mean[:, sp_dim] = scaler_fp.inverse_transform(output_mean[:, sp_dim].reshape(-1, 1)).flatten()
        output_std[:, sp_dim] *= scaler_fp.scale_[0]
        if clip_positive:
            output_mean = np.clip(output_mean, 0, None)
        elapsed = time.time() - start_time
        print(f'##### Prediction Done in {elapsed:.2f} seconds #####')
    else:
        input_xy= [ xy_pred[measured_s].copy() for _ in range(sp.shape[1])]+[xy_pred[measured_f]]
        output_z= [sp[:, i][:,None] for i in range(sp.shape[1])]+ [fp]
        output_mean=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        output_std=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        if matrix=='LCM':
            K = GPy.kern.Matern32(2)
            K.lengthscale.constrain_bounded(0.01, 2.0)
            K.variance.constrain_bounded(0.01, 2.0)
            lcm = GPy.util.multioutput.LCM(input_dim=2,num_outputs=sp.shape[1]+fp.shape[1],kernels_list=[K], W_rank=W)
            m = GPy.models.GPCoregionalizedRegression(input_xy, output_z,kernel=lcm)
            m['.*ICM.*var'].unconstrain()
            m['.*ICM.*W'].unconstrain()
            m.optimize(max_iters=5000)
        elif matrix =='ICM':
            K = GPy.kern.Matern32(2)
            K.lengthscale.constrain_bounded(0.01, 2.0)
            K.variance.constrain_bounded(0.01, 2.0)
            icm = GPy.util.multioutput.ICM(input_dim=2, num_outputs=sp.shape[1]+fp.shape[1], kernel=K, W_rank=W)
            m = GPy.models.GPCoregionalizedRegression(input_xy, output_z, kernel=icm)
            m['.*ICM.*var'].unconstrain()
            m['.*ICM.*W'].unconstrain()
            m.optimize(max_iters=5000)
        if gp_verbose:
            print(m)
        
        for i in range(sp.shape[1]+fp.shape[1]):
            newX = np.hstack([xy_pred, np.ones((xy_pred.shape[0], 1)) * i])  # shape (177, 2)
            noise_dict = {'output_index': newX[:, 2:].astype(int)}  # shape (177, 1)
            m_, v_ = m.predict(newX, Y_metadata=noise_dict)  # m_: shape (177, 1)
            output_mean[:,i]= m_.flatten()
            output_std[:,i]= v_.flatten()
        if clip_positive:
            output_mean = np.clip(output_mean, 0, None)
        elapsed = time.time() - start_time
        print(f'##### Prediction Done in {elapsed:.2f} seconds #####')

    return output_mean, output_std, model_dict

def independent_GP(xy_pred: np.array, measured_f: np.array, measured_s: np.array, fp: np.array, sp: np.array, 
                            method="ICM", W=1, scale=True, clip_positive=True, gp_verbose=True)-> tuple:
    """
    Use LCM but set the B.W to be 0 and let the kappa free
    regression input and output 
    Take phase abundances and functional property as input 
    :xy: 2d coordinates 
    :measured_f: index 
    :measured_s: index 
    :fp: (number of sample, 1) -> resistance 
    :sp: (number of sample, number of component) -> abundance
    :W: how complex the pattern of correlation
    :scale: whether to scale data or not 
    :clip_positive: to avoid negative output prediction, to fit NMF framework 
    return mean, std, gp_dict
    """ 
    start_time = time.time()
    
    if scale:
        scaler_fp = StandardScaler()
        scaler_sp = StandardScaler()
        fp_scaled = scaler_fp.fit_transform(fp)
        sp_scaled = scaler_sp.fit_transform(sp)
        input_xy= [ xy_pred[measured_s].copy() for _ in range(sp.shape[1])]+[xy_pred[measured_f]]
        output_z= [sp_scaled[:, i][:,None] for i in range(sp.shape[1])]+ [fp_scaled]
        output_mean=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        output_std=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        
       
        K = GPy.kern.Matern32(2)
        K.lengthscale.constrain_bounded(0.01, 2.0)
        K.variance.constrain_bounded(0.01, 2.0)
        icm = GPy.util.multioutput.ICM(input_dim=2, num_outputs=sp.shape[1]+fp.shape[1], kernel=K, W_rank=W)
        icm.B.W.constrain_fixed(0)
        icm.B.kappa.unconstrain() 
        m = GPy.models.GPCoregionalizedRegression(input_xy, output_z, kernel=icm)
        m['.*ICM.*var'].unconstrain()
        m['.*ICM.*W'].unconstrain()
        m.optimize(max_iters=5000)
        if gp_verbose:
            print(m)
        model_dict = m.to_dict(save_data=True)

        for i in range(sp.shape[1]+fp.shape[1]):
            newX = np.hstack([xy_pred, np.ones((xy_pred.shape[0], 1)) * i])  # shape (177, 2)
            noise_dict = {'output_index': newX[:, 2:].astype(int)}  # shape (177, 1)
            m_, v_ = m.predict(newX, Y_metadata=noise_dict)  # m_: shape (177, 1)
            output_mean[:,i]= m_.flatten()
            output_std[:,i]= v_.flatten()
        
        sp_dim = sp.shape[1]
        output_mean[:, :sp_dim] = scaler_sp.inverse_transform(output_mean[:, :sp_dim]) #fix negative prediction although very samll 
        output_std[:, :sp_dim] *= scaler_sp.scale_
        output_mean[:, sp_dim] = scaler_fp.inverse_transform(output_mean[:, sp_dim].reshape(-1, 1)).flatten()
        output_std[:, sp_dim] *= scaler_fp.scale_[0]
        if clip_positive:
            output_mean = np.clip(output_mean, 0, None)
        elapsed = time.time() - start_time
        
        print(f'##### Prediction Done in {elapsed:.2f} seconds #####')
    else:
        
        input_xy= [ xy_pred[measured_s].copy() for _ in range(sp.shape[1])]+[xy_pred[measured_f]]
        output_z= [sp[:, i][:,None] for i in range(sp.shape[1])]+ [fp]
        output_mean=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        output_std=np.zeros((xy_pred.shape[0],sp.shape[1]+fp.shape[1]))
        

        K = GPy.kern.Matern32(2)
        K.lengthscale.constrain_bounded(0.01, 2.0)
        K.variance.constrain_bounded(0.01, 2.0)
        icm = GPy.util.multioutput.ICM(input_dim=2, num_outputs=sp.shape[1]+fp.shape[1], kernel=K, W_rank=W)
        icm.B.W.constrain_fixed(0)
        icm.B.kappa.unconstrain() 
        m = GPy.models.GPCoregionalizedRegression(input_xy, output_z, kernel=icm)
        m['.*ICM.*var'].unconstrain()
        m['.*ICM.*W'].unconstrain()
        m.optimize(max_iters=5000)
        print(m)
        model_dict = m.to_dict(save_data=True)
        
        for i in range(sp.shape[1]+fp.shape[1]):
            newX = np.hstack([xy_pred, np.ones((xy_pred.shape[0], 1)) * i])  # shape (177, 2)
            noise_dict = {'output_index': newX[:, 2:].astype(int)}  # shape (177, 1)
            m_, v_ = m.predict(newX, Y_metadata=noise_dict)  # m_: shape (177, 1)
            output_mean[:,i]= m_.flatten()
            output_std[:,i]= v_.flatten() # variance 
        if clip_positive:
            output_mean = np.clip(output_mean, 0, None)
        elapsed = time.time() - start_time
        print(f'##### Prediction Done in {elapsed:.2f} seconds #####')

    return output_mean, output_std, model_dict