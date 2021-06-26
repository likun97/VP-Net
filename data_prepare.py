# something about making dataset 






def single_mat_64_no_down(used_ref, used_ms, used_pan):
    
    
# ###    '''normalization'''  TGARS所有数据都是自己一手做出来的 已经在matlab里标准化   这里不再需要标准化   
# =============================================================================
#     max_patch, min_patch = np.max(used_ref, axis=(0,1)), np.min(used_ref, axis=(0,1))
#     used_ref = np.float32(used_ref-min_patch) / (max_patch - min_patch)

#     max_patch, min_patch = np.max(used_ms, axis=(0,1)), np.min(used_ms, axis=(0,1))
#     used_ms = np.float32(used_ms-min_patch) / (max_patch - min_patch)
#     max_patch, min_patch = np.max(used_pan, axis=(0,1)), np.min(used_pan, axis=(0,1))
#     used_pan = np.float32(used_pan-min_patch) / (max_patch - min_patch) 
     
# =============================================================================

    used_ref       = used_ref    # used_hrhs 
    downgrade_MS   = used_ms     # used_lrhs
    downgrade_PAN  = used_pan    # used_hrms
        
    train_original_used_MS = []   
    train_downgrade_PAN    = []   
    train_downgrade_MS     = []
    
###    """crop images""" 
    stride        = 24
    training_size = 64  
    
    for j in range(0, downgrade_PAN.shape[0]-training_size +1, stride):
        for k in range(0, downgrade_PAN.shape[1]-training_size +1, stride):
            
            temp_hrhs = used_ref      [j:j+training_size,     k:k+training_size,  :]          
            temp_hrms = downgrade_PAN[j:j+training_size,     k:k+training_size,  :]
#            temp_lrhs = downgrade_MS [j:j+training_size,     k:k+training_size,  :]                               #  对比  PanNet  
            temp_lrhs = downgrade_MS [int(j/4):int((j+training_size)/4),  int(k/4):int((k+training_size)/4), :]
            
            
            train_original_used_MS.append(temp_hrhs)
            train_downgrade_PAN   .append(temp_hrms)
            train_downgrade_MS    .append(temp_lrhs)
            
    train_original_used_MS = np.array(train_original_used_MS, dtype='float32')
    train_downgrade_PAN    = np.array(train_downgrade_PAN,    dtype='float32')
    train_downgrade_MS     = np.array(train_downgrade_MS,     dtype='float32') 
    
    return train_original_used_MS, train_downgrade_PAN, train_downgrade_MS



# ======================================================================================================= 
import os
import numpy as np
import scipy.io as sio

used_ms   = []
used_pan  = []
used_ref  = []

path  = 'E:\datasets\\4_GF1_GF2\\GF2\\crop_xj_smooth_down\\mat60'   

file_list = os.listdir(path)  
file_list.sort(key=lambda x:int(x.split('.')[0].split('GF2_300_')[1])) 

for file in file_list:                                                         
    if not os.path.isdir(file):                                                
        mat_data = sio.loadmat(path+"/"+file)                                  
        
        mat_ms   = mat_data['I_MS_LR']   
        used_ms.append(mat_ms)
        
        mat_pan   = mat_data['I_PAN_LR']  
        mat_pan   = np.expand_dims(mat_pan,-1)          
        used_pan.append(mat_pan)
        
        mat_ref  = mat_data['Ref'] 
        used_ref.append(mat_ref)  
        # used_pan  = np.vstack((used_pan,mat_pan)) 

print('used_ms.len' ,len(used_ms)) 
print('used_pan.len',len(used_pan)) 
print('used_ref.len' ,len(used_ref)) 
 

original_used_MS, downgrade_PAN, downgrade_MS  =  single_mat_64_no_down  (used_ref[0],used_ms[0], used_pan[0] )              

for i in range(1, len(used_pan)):
    for_original_used_MS, for_downgrade_PAN, for_downgrade_MS = single_mat_64_no_down (used_ref[i], used_ms[i], used_pan[i] )  
                                                                        
    original_used_MS =  np.concatenate( (original_used_MS, for_original_used_MS) ,axis = 0)
    downgrade_PAN    =  np.concatenate( (downgrade_PAN,    for_downgrade_PAN) ,axis = 0)
    downgrade_MS     =  np.concatenate( (downgrade_MS,     for_downgrade_MS) ,axis = 0)                   
 
import random
index = [i for i in range(original_used_MS.shape[0])]
random.shuffle(index)

random.shuffle(index)
train_original_used_MS = original_used_MS [index, :, :, :]
train_downgrade_PAN    = downgrade_PAN [index, :, :, :]
train_downgrade_MS     = downgrade_MS [index, :, :, :]
 
sio.savemat('E:\GF2_train_from_single_pic_no_down_no_normal_60.mat', dict([('ms', train_downgrade_MS), ('pan', train_downgrade_PAN),('ref',train_original_used_MS)]) )


########################################################################################################################################
########################################################################################################################################

def prepare_input_64(used_ms, used_pan):
    
###    '''normalization'''
    max_patch, min_patch = np.max(used_ms, axis=(0,1)), np.min(used_ms, axis=(0,1))
    used_ms = np.float32(used_ms-min_patch) / (max_patch - min_patch)
    max_patch, min_patch = np.max(used_pan, axis=(0,1)), np.min(used_pan, axis=(0,1))
    used_pan = np.float32(used_pan-min_patch) / (max_patch - min_patch)
    
    print('after normalization:')
    print('used_pan.shape' ,used_pan.shape,'used_ms.shape'  ,used_ms.shape) 
    print('\n') 
    
###############################################################################
    
###    '''downgrade'''
    ratio=4

    
    downgrade_MS ,downgrade_PAN = downgrade_images(used_ms, used_pan, ratio, sensor= None)

    print('after normalization:')                            
    print('downgrade_PAN.shape',downgrade_PAN.shape,'downgrade_MS.shape' ,downgrade_MS.shape)
###############################################################################

 
###    """crop images""" 
    stride        = 8
    training_size = 64                   
    
    used_ms        = used_ms          # used_hrhs 
    downgrade_MS   = downgrade_MS     # used_lrhs
    downgrade_PAN  = downgrade_PAN    # used_hrms
        
    train_original_used_MS = []   
    train_downgrade_PAN    = []   
    train_downgrade_MS     = []
    
    for j in range(0, downgrade_PAN.shape[0]-training_size + stride, stride):
        for k in range(0, downgrade_PAN.shape[1]-training_size + stride, stride):
            
            temp_hrhs = used_ms      [j:j+training_size,     k:k+training_size,  :]          
            temp_hrms = downgrade_PAN[j:j+training_size,     k:k+training_size,  :]
#            temp_lrhs = downgrade_MS [j:j+training_size,     k:k+training_size,  :]                               #  对比  PanNet  
            temp_lrhs = downgrade_MS [int(j/4):int((j+training_size)/4),  int(k/4):int((k+training_size)/4), :]
            
            
            train_original_used_MS.append(temp_hrhs)
            train_downgrade_PAN   .append(temp_hrms)
            train_downgrade_MS    .append(temp_lrhs)
            
    train_original_used_MS = np.array(train_original_used_MS, dtype='float32')
    train_downgrade_PAN    = np.array(train_downgrade_PAN,    dtype='float32')
    train_downgrade_MS     = np.array(train_downgrade_MS,     dtype='float32')
    
    index = [i for i in range(train_original_used_MS.shape[0])]
    random.shuffle(index)

    random.shuffle(index)
    train_original_used_MS = train_original_used_MS[index, :, :, :]
    train_downgrade_PAN = train_downgrade_PAN[index, :, :, :]
    train_downgrade_MS = train_downgrade_MS[index, :, :, :]
    
    print('after croping:')
    print(' train_original_used_MS.shape',train_original_used_MS.shape)
    print(' train_downgrade_PAN.shape'   ,train_downgrade_PAN.shape)
    print(' train_downgrade_MS.shape'    ,train_downgrade_MS.shape)
    
    print('\n')
    
    return train_original_used_MS, train_downgrade_PAN, train_downgrade_MS

           
        
