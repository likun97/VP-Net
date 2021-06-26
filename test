# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:43:18 2021

@author: KunLi
"""
 
import os
import numpy as np
import scipy.io as sio

# ============================================================================= 
# ms_path  = 'E:\datasets\\3_QB-Wuhan\\crop_xj_30_01_smooth_down\mat10'                                                   
# ms_file_list = os.listdir(ms_path)                                                        
# ms_file_list.sort(key=lambda x:int(x.split('.')[0].split('QB_WH_down_')[1]))          
 
# used_ms   = []
# used_pan  = []
# used_ref  = []

# for file in ms_file_list:                                                                   
#     if not os.path.isdir(file):                                                             
#         mat_data = sio.loadmat(ms_path+"/"+file)                                           
        
#         mat_ms   = mat_data['I_MS_LR']   
#         used_ms.append(mat_ms)
        
#         mat_pan   = mat_data['I_PAN_LR'] 
#         used_pan.append(mat_pan) 
        
#         mat_ref   = mat_data['Ref'] 
#         used_ref.append(mat_ref) 
        
# ============================================================================= 

# ms_path  = 'E:\\datasets\\4_GF1_GF2\GF2\\crop_xj_smooth_down\\mat161_304_for_test'  

# we give 5 test examples  (from GF2 satellite)

ms_path  = './data'                                  
ms_file_list = os.listdir(ms_path)                                                          
ms_file_list.sort(key=lambda x:int(x.split('.')[0].split('GF2_300_')[1]))        
 
used_ms   = []
used_pan  = []
used_ref  = []

for file in ms_file_list:                                                                  
    if not os.path.isdir(file):                                                             
        mat_data = sio.loadmat(ms_path+"/"+file)                                           
        
        mat_ms   = mat_data['I_MS_LR']   
        used_ms.append(mat_ms)
        
        mat_pan   = mat_data['I_PAN_LR'] 
        used_pan.append(mat_pan) 
        
        mat_ref   = mat_data['Ref'] 
        used_ref.append(mat_ref) 
        
        
# ===============================================================================================     
# ===============================================================================================  
model_path = './checkpoint/'            
save_path  = 'test_1/'    # provide a folder name to save test result according to test dataset

import os
import cv2
import numpy as np
import scipy.io as sio    
import tensorflow as tf

from utils        import downgrade_images
from fusion_net   import vp_net, compute_cost   
from metrics      import ref_evaluate , no_ref_evaluate                     
     


tf.reset_default_graph() 
PhaseNumber =7
test_label = np.zeros((300, 300, 4), dtype = 'float32')

X_output = tf.placeholder(tf.float32, shape=(1, 300,    300,    4))                       
P_input  = tf.placeholder(tf.float32, shape=(1, 300,    300,    1))                           
M_input  = tf.placeholder(tf.float32, shape=(1, 75,     75,     4))  


PredX , ListX  ,Q  = vp_net( PhaseNumber, M_input, P_input, X_output )           

config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep = 5)

import time
time_all =[]
with tf.Session(config=config) as sess:  
      
   ckpt = tf.train.latest_checkpoint(model_path)
   saver.restore(sess, ckpt) 
   
   
   # for num in range(5): 
   for num in range(len(used_pan)):

       HR_Ref  =  used_ref[num]                     # GT  真值   全都已经归一化 下采样了
       
       LR_ms   =  used_ms[num]
       LR_pan  =  used_pan[num]     
       LR_pan  = np.expand_dims(LR_pan, -1) 
       
       LR_ms_test  = np.expand_dims(LR_ms, 0)
       LR_pan_test = np.expand_dims(LR_pan, 0)
       
       batch_M = LR_ms_test 
       batch_P = LR_pan_test
       
       time_start = time.time() 

       one , _ = sess.run([PredX[-1], ListX[-1]], feed_dict={M_input: batch_M,  P_input: batch_P} )
 
   
       time_end = time.time()     
       time_c   = time_end - time_start   
       print('time cost', time_c, 's')   
       
       time_all.append(time_c) 
       
       one = np.clip(one, 0, 1)   
       
       test_label = one[0,:,:,:]
       print('test_label',test_label.shape)
        
       save_testimage_dir='./test_imgs/' +save_path

       save_test_mat_dir='./test_mats/' +save_path

       if not os.path.exists(save_testimage_dir):
           os.makedirs(save_testimage_dir)
       if not os.path.exists(save_test_mat_dir):
           os.makedirs(save_test_mat_dir)
    
       cv2.imwrite (save_testimage_dir +'%d_test.png'%(num+1) ,np.uint8(255*test_label)[:, :, [0,1,2]] )    
       cv2.imwrite (save_testimage_dir +'%d_ms.png'%(num+1) ,  np.uint8(255*HR_Ref)    [:, :, [0,1,2]] ) 
       
        # save mat
       sio.savemat (save_test_mat_dir  +'Variation_%d.mat'%(num+1), { 'ref':np.uint8(255*HR_Ref), 'fusion':np.uint8(255*test_label)} )
       
       
        
       
       gt = HR_Ref 
       
       ref_results={}
       ref_results.update({'metrics: ':'  PSNR,   SSIM,   SAM,   ERGAS,  SCC,     Q,     RMSE'})
       no_ref_results={}
       no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})
       
       temp_ref_results    = ref_evaluate( np.uint8(255*test_label), np.uint8(255*HR_Ref) )   
       temp_no_ref_results = no_ref_evaluate( test_label,  LR_pan ,  LR_ms )    
        
       ref_results   .update({'xxx     ':temp_ref_results})
       no_ref_results.update({'xxx     ':temp_no_ref_results})
       
       save_testlog_dir='./test_logs/' + save_path 
       if not os.path.exists(save_testlog_dir):
           os.makedirs(save_testlog_dir)
       lk_output_file_ref    = save_testlog_dir+"ref_test_1.txt"              
       lk_output_file_no_ref = save_testlog_dir+"no_ref_test_1.txt"  
       
      
       print('################## reference  #######################')
       for index, i in enumerate(ref_results):
           if index == 0:
               print(i, ref_results[i])
       else:    
               print(i, [round(j, 4) for j in ref_results[i]])
               
               
               list2str= str([ round(j, 4) for j in ref_results[i] ])
               list2str= ('%d  '+ list2str+'\n')%(num+1) 
               lk_output_file = open(lk_output_file_ref, 'a')
               lk_output_file.write(list2str)
               lk_output_file.close()  
       
       print('################## no reference  ####################')
      
       for index, i in enumerate(no_ref_results):
            if index == 0:
                print(i, no_ref_results[i])
            else:    
                print(i, [round(j, 4) for j in no_ref_results[i]])
               
               
                list2str= str([ round(j, 4) for j in no_ref_results[i] ])     
                list2str=('%d  '+ list2str+'\n')%(num+1) 
                lk_output_file = open(lk_output_file_no_ref, 'a')
                lk_output_file.write(list2str)
                lk_output_file.close()  
       print('#####################################################')
       
                
       
     
   print('test finished')
 
