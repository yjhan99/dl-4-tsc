
# coding: utf-8

# # Data augmentation for time-series data

# #### This is a simple example to apply data augmentation to time-series data (e.g. wearable sensor data). If it helps your research, please cite the below paper. 

# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220. 

# https://dl.acm.org/citation.cfm?id=3136817
# 
# https://arxiv.org/abs/1706.00527

# @inproceedings{TerryUm_ICMI2017,
#  author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
#  title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
#  booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
#  series = {ICMI 2017},
#  year = {2017},
#  isbn = {978-1-4503-5543-8},
#  location = {Glasgow, UK},
#  pages = {216--220},
#  numpages = {5},
#  doi = {10.1145/3136755.3136817},
#  acmid = {3136817},
#  publisher = {ACM},
#  address = {New York, NY, USA},
#  keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
# } 

# #### You can freely modify this code for your own purpose. However, please leave the above citation information untouched when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Terry Taewoong Um (terry.t.um@gmail.com)
# 
# https://twitter.com/TerryUm_ML
# 
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data



import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline      # for warping
# from transforms3d.axangles import axangle2mat  # for rotation


class DataAugmentation:
    def __init__(self, myX):
        self.myX = myX.reshape(-1,1)

    # ## 1. Jittering
    # #### Hyperparameters :  sigma = standard devitation (STD) of the noise
    sigma = 0.05
    def DA_Jitter(self, X, sigma=0.05):
        myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        return X+myNoise

    # ## 2. Scaling
    # #### Hyperparameters :  sigma = STD of the zoom-in/out factor
    sigma = 0.1
    def DA_Scaling(self, X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
        myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
        return X*myNoise

    # ## 3. Magnitude Warping
    # #### Hyperparameters :  sigma = STD of the random knots for generating curves
    # #### knot = # of knots for the random curves (complexity of the curves)
    # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample". 
    # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
    sigma = 0.2
    knot = 4
    ## This example using cubic splice is not the best approach to generate random curves. 
    ## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
    @staticmethod
    def GenerateRandomCurves(X, sigma=0.2, knot=4):
        xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
        x_range = np.arange(X.shape[0])
        cs_x = CubicSpline(xx[:,0], yy[:,0])
        # cs_y = CubicSpline(xx[:,1], yy[:,1])
        # cs_z = CubicSpline(xx[:,2], yy[:,2])
        # return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()
        return np.array([cs_x(x_range)]).transpose()

    def DA_MagWarp(self, X, sigma):
        return X * self.GenerateRandomCurves(X, sigma)


    # ## 4. Time Warping
    # #### Hyperparameters :  sigma = STD of the random knots for generating curves
    # #### knot = # of knots for the random curves (complexity of the curves)
    sigma = 0.2
    knot = 4
    def DistortTimesteps(self, X, sigma=0.2):
        tt = self.GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        # t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
        t_scale = [(X.shape[0]-1)/tt_cum[-1,0]]
        tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
        # tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
        # tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
        return tt_cum


    def DA_TimeWarp(self, X, sigma=0.2):
        tt_new = self.DistortTimesteps(X, sigma)
        X_new = np.zeros(X.shape)
        x_range = np.arange(X.shape[0])
        X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
        # X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
        # X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
        return X_new

    # ## 5. Rotation
    # #### Hyperparameters :  N/A
    @staticmethod
    def DA_Rotation(X):
        # axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
        # angle = np.random.uniform(low=-np.pi, high=np.pi)
        # return np.matmul(X , axangle2mat(axis,angle))
        return -X

    # ## 6. Permutation
    # #### Hyperparameters :  nPerm = # of segments to permute
    # #### minSegLength = allowable minimum length for each segment
    nPerm = 4
    minSegLength = 100
    @staticmethod
    def DA_Permutation(X, nPerm=4, minSegLength=10):
        X_new = np.zeros(X.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
            segs[-1] = X.shape[0]
            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
            X_new[pp:pp+len(x_temp),:] = x_temp
            pp += len(x_temp)
        return(X_new)


    # ## 7. Random Sampling
    # #### Hyperparameters :  nSample = # of subsamples (nSample <= X.shape[0])
    # This approach is similar to TimeWarp, but will use only subsamples (not all samples) for interpolation. (Using TimeWarp is more recommended)
    nSample = 1000
    @staticmethod
    def RandSampleTimesteps(X, nSample=1000):
        X_new = np.zeros(X.shape)
        tt = np.zeros((nSample,X.shape[1]), dtype=int)
        tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        # tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        # tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
        tt[-1,:] = X.shape[0]-1
        return tt

    def DA_RandSampling(self, X, nSample=1000):
        tt = self.RandSampleTimesteps(X, nSample)
        X_new = np.zeros(X.shape)
        X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
        # X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
        # X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
        return X_new
    

    def apply_all_augmentations(self):
        # print(self.myX.shape)
        jittered_data = self.DA_Jitter(self.myX)
        scaled_data = self.DA_Scaling(self.myX)
        mag_warped_data = self.DA_MagWarp(self.myX, sigma=0.2)
        time_warped_data = self.DA_TimeWarp(self.myX, sigma=0.2)
        rotated_data = self.DA_Rotation(self.myX)
        permuted_data = self.DA_Permutation(self.myX)
        rand_sampled_data = self.DA_RandSampling(self.myX)

        augmented_data = np.concatenate([jittered_data, scaled_data, mag_warped_data,
                                         time_warped_data, rotated_data, permuted_data,
                                         rand_sampled_data], axis=0)
        augmented_data = augmented_data.reshape(-1)
        
        # print(augmented_data.shape)

        return augmented_data