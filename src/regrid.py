import numpy as np
import skimage.measure

def cal_sigma(res, w):
    # original data is 1 km 
    # calculate the sigma in special resolution (res)
    # for example res = 16 means 16km resolution
    greater_w = w > 0.5
    sigma = np.zeros((w.shape[0], int(w.shape[1]/res), int(w.shape[2]/res)))
    for t in range(w.shape[0]):
        sigma[t] = skimage.measure.block_reduce(greater_w[t], (res,res), np.mean)

    return sigma


def feature_regrid(res, array):
    # regrid data to different resolution 
    # input array is feature you want to regrid
    out = np.zeros((array.shape[0], int(array.shape[1]/res), int(array.shape[2]/res)))
    for t in range(array.shape[0]):
        out[t] = skimage.measure.block_reduce(array[t], (res,res), np.mean)
    print(out.shape)
    return out

if __name__ == '__main__':

    # load data
    z = np.loadtxt('../data/z.txt')
    w = np.load('../data/w.npy')[:,:,:] # for calculate sigma
    u = np.load('../data/u.npy')[:,:,:]
    v = np.load('../data/v.npy')[:,:,:]
    qv = np.load('../data/qv.npy')[:,:,:]
    th = np.load('../data/th.npy')[:,:,:]
    
    # feature regrid
    u_d8 = feature_regrid(9, u)
    u_d16 = feature_regrid(15, u)
    u_d32 = feature_regrid(45, u)
    u_d64 = feature_regrid(135, u)

    v_d8 = feature_regrid(9, v)
    v_d16 = feature_regrid(15, v)
    v_d32 = feature_regrid(45, v)
    v_d64 = feature_regrid(135, v)
    
    qv_d8 = feature_regrid(9, qv)
    qv_d16 = feature_regrid(15, qv)
    qv_d32 = feature_regrid(45, qv)
    qv_d64 = feature_regrid(135, qv)
    
    th_d8 = feature_regrid(9, th)
    th_d16 = feature_regrid(15, th)
    th_d32 = feature_regrid(45, th)
    th_d64 = feature_regrid(135, th)
    
    # sigma regrid
    sigma_d8 = cal_sigma(9, w)
    sigma_d16 = cal_sigma(15, w)
    sigma_d32 = cal_sigma(45, w)
    sigma_d64 = cal_sigma(135, w)


    # save data

    np.save('../data/d9_sigma.npy', sigma_d8)
    np.save('../data/d15_sigma.npy', sigma_d16)
    np.save('../data/d45_sigma.npy', sigma_d32)
    np.save('../data/d135_sigma.npy', sigma_d64)

    np.save('../data/d9_u.npy', u_d8)
    np.save('../data/d15_u.npy', u_d16)
    np.save('../data/d45_u.npy', u_d32)
    np.save('../data/d135_u.npy', u_d64)
    
    np.save('../data/d9_v.npy', v_d8)
    np.save('../data/d15_v.npy', v_d16)
    np.save('../data/d45_v.npy', v_d32)
    np.save('../data/d135_v.npy', v_d64)
    
    np.save('../data/d9_qv.npy', qv_d8)
    np.save('../data/d15_qv.npy', qv_d16)
    np.save('../data/d45_qv.npy', qv_d32)
    np.save('../data/d135_qv.npy', qv_d64)
    
    np.save('../data/d9_th.npy', th_d8)
    np.save('../data/d15_th.npy', th_d16)
    np.save('../data/d45_th.npy', th_d32)
    np.save('../data/d135_th.npy', th_d64)
