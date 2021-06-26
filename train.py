# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:58:09 2021

@author: kunLi
"""
# -*- coding: utf-8 -*-
 


import os
import numpy as np
import scipy.io as sio 

#  First, you need to make your own training dataset  ---- 

GF2_train_from_single_pic160  = sio.loadmat('E:\datasets\\4_GF1_GF2\\GF2\\crop_xj_smooth_down\\mat160_to_train_set\\GF2_train_from_single_pic_no_down_no_normal.mat')
train_original_used_MS       = GF2_train_from_single_pic160['ref']
train_downgrade_PAN          = GF2_train_from_single_pic160['pan']
train_downgrade_MS           = GF2_train_from_single_pic160['ms']



#%%

from fusion_net import vp_net, compute_cost  
import tensorflow as tf

import skimage.measure
import time

  
nrtrain       = 16000    
EpochNum      = 55     
batch_size    = 40
PhaseNumber   = 7 
learning_rate = 0.0001
   
 
tf.reset_default_graph()                                                        

X_output = tf.placeholder(tf.float32, shape=(batch_size, 64,    64,    4))                     
P_input  = tf.placeholder(tf.float32, shape=(batch_size, 64,    64,    1))              
M_input  = tf.placeholder(tf.float32, shape=(batch_size, 16,    16,    4))                   

                                                                                        

PredX , ListX ,Q = vp_net( PhaseNumber, M_input, P_input, X_output )
cost  , cost_sym = compute_cost ( PredX, ListX ,P_input ,X_output, PhaseNumber)      

 
cost_all    = 10*cost  + 0.1*cost_sym      
optm_all    = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all) 
 
saver  = tf.train.Saver(tf.global_variables(), max_to_keep=100)
tf.ConfigProto().gpu_options.allow_growth = True 
time_start = time.time() 

      
  

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
 
    print('#'*60," Strart Training... ")
 

    model_dir           = 'VP_Net_Phase%d_epoch%d_GF2_Model' % (PhaseNumber,EpochNum)                         
    output_file_name    = "./train_log/%s_log.txt" % (model_dir)
    lk_output_file_name = "./train_log/batch_%s_log.txt" % (model_dir)

    for epoch_i in range(0, EpochNum):                                          
    
        print('##############')
        print('Training with %d epoch,  learning rate =%.5f'%(epoch_i+1, learning_rate)) 
        
        Training_Loss = 0 
        randidx_all   = np.random.permutation(nrtrain)
                                                                               
        for batch_i in range(nrtrain // batch_size):                            
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

            batch_X = train_original_used_MS[randidx, :, :, :] 
            batch_P = train_downgrade_PAN   [randidx, :, :, :] 
            batch_M = train_downgrade_MS    [randidx, :, :, :]
 
            feed_dict = { M_input: batch_M, P_input: batch_P, X_output: batch_X }
        
            _ , cost_all_value = sess.run([optm_all, cost_all], feed_dict = feed_dict) 
            Training_Loss     += cost_all_value 

        # visual output 
            _ ,ifshow = divmod(batch_i+1,300) 
            if ifshow ==1:
                P_PredX , P_ListX = sess.run([PredX[-1], ListX[-1]],feed_dict={M_input: batch_M,  P_input: batch_P} )   
   
                print('PredX[-1].shape, ListX[-1].shape', P_PredX.shape, P_ListX.shape)        
                                                                                             
        # eval this batch      
                psnr     = skimage.metrics.peak_signal_noise_ratio(batch_X, P_PredX )        
                ssim     = skimage.metrics.structural_similarity  (batch_X, P_PredX, multichannel=True)
                nrmse    = skimage.metrics.normalized_root_mse    (batch_X, P_PredX )
                mse      = skimage.metrics.mean_squared_error     (batch_X, P_PredX)
                 
                CurLoss  = Training_Loss/(batch_i+1)
                
                print       ('In %d epoch %d-th batch , Training_Loss =%.8f, PSNR =%.3f, SSIM =%.4f, NRMSE =%.5f\n' %(epoch_i+1, batch_i+1, CurLoss,  psnr, ssim, nrmse))
                write_data = 'In %d epoch i-th batch , Training_Loss =%.8f, PSNR =%.8f, SSIM =%.8f, NRMSE =%.8f\n' %(epoch_i+1, CurLoss,  psnr, ssim, nrmse)
                
                out_file = open(lk_output_file_name, 'a')
                out_file.write(write_data)
                out_file.close()
        enter = open(lk_output_file_name, 'a')
        enter.write('\n')
        enter.close()
        
        
        output_data = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch_i+1, EpochNum, sess.run(cost, feed_dict=feed_dict), sess.run(cost_sym, feed_dict=feed_dict))
        print('##############')
        print(output_data)

        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        
# save model
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        if epoch_i <= 10:
            saver.save(sess, './train_model/%s/VP_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=True)
        else:
            if epoch_i % 5 == 0:
                saver.save(sess, './train_model/%s/VP_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)


    print("Training Finished")
    sess.close()
      
time_end = time.time()    
time_c   = time_end - time_start     

