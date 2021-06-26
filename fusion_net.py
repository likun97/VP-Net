# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:52:20 2021

@author: KunLi
""" 
    
#%%
import tensorflow as tf
import numpy as np

                     
def vp_net( n , M_input, P_input, X_output):

    layers_predict    = []                                         
    layers_symetric   = []                                      
    layers_variable_Q = [] 
 
    X0  = tf.image.resize_images(M_input, size=[M_input.shape[1]*4,M_input.shape[2]*4], method=0) 
    X0_av_single = tf.reduce_mean(X0, axis=3)
    X0_av_copy   = tf.tile( tf.expand_dims(X0_av_single,axis=-1) , multiples=[1, 1, 1, 4])       
    Q0           = X0_av_copy    

    layers_predict.append(X0)  
    layers_variable_Q.append(Q0)  
    
    for i in range(n):
        with tf.variable_scope('conv_%d' %i):     
            [pred, sym, Q] = ista_block (layers_predict, layers_variable_Q, M_input, P_input, X_output, i)   
                                   
            layers_predict   .append(pred)
            layers_symetric  .append(sym)  
            layers_variable_Q.append(Q)                        
            
    return layers_predict, layers_symetric, layers_variable_Q

 
 
def ista_block(pred_layers, Q_layers, M_input, P_input, X_output, layer_no):

    filter_size = 3
    conv_size   = 40
    step_coffi  = tf.Variable(0.1, dtype=tf.float32) 
    mue_coffi   = tf.Variable(0.05, dtype=tf.float32) 
    soft_thr    = tf.Variable(0.1, dtype=tf.float32) 
    
    [Weights_, bias_]   = add_con2d_weight_bias([filter_size, filter_size, 4, conv_size]        , [conv_size], 4)

    [Weights0, bias0]   = add_con2d_weight_bias([filter_size, filter_size, 1, conv_size]        , [conv_size], 0)   
    [Weights1, bias1]   = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)  
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11) 
    [Weights2, bias2]   = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)  
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22) 
    [Weights3, bias3]   = add_con2d_weight_bias([filter_size, filter_size, conv_size, 4]        , [4]        , 3)  
    
    
    
    
    down_input            = tf.image.resize_images(pred_layers[-1], size=[(pred_layers[-1].shape[1])//4,(pred_layers[-1].shape[2])//4], method=0) 
    down_input_M_input    = down_input - M_input 
    UP_down_input_M_input = tf.image.resize_images(down_input_M_input, size=[(down_input_M_input.shape[1])*4,(down_input_M_input.shape[2])*4], method=0)
    UP_down_input_M_input = tf.scalar_mul(step_coffi, UP_down_input_M_input ) 
    
    r_k  = pred_layers[-1] - UP_down_input_M_input
    
    
    up_M         = tf.image.resize_images(M_input, size=[(M_input.shape[1])*4,(M_input.shape[2])*4], method=0) 
    up_M_av      = tf.reduce_mean(up_M, axis=3)
    up_M_av_copy = tf.tile( tf.expand_dims(up_M_av,axis=-1) , multiples=[1, 1, 1, 4]) 
    
    
    R_T    =  tf.div(up_M , tf.add(up_M_av_copy, 0.1) )                             
    R_T_Q  =  tf.multiply(Q_layers[-1] , R_T)   
    x_k    = tf.div( r_k + tf.scalar_mul(mue_coffi, R_T_Q) , (1 + mue_coffi))          
  
    
    R     = tf.div(up_M_av_copy , tf.add(up_M, 0.1)  )                              
    R_X   = tf.multiply(x_k , R)  


  # % % %  % %  % % %  % % % % %  % % % % %  % % % % %  % % % % %  % % % % %  %   
  
    x3_ista   = tf.nn.conv2d(R_X, Weights_, strides=[1, 1, 1, 1], padding='SAME')                    #          —— Weights0   ——>>  R_X   to use 
    x4_ista   = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))    #          —— Weights1 
    x44_ista  = tf.nn.conv2d(x4_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')               #          —— Weights11  ——>>  F(R_X) 
    
    xp_ista   = tf.nn.conv2d(P_input, Weights0, strides=[1, 1, 1, 1], padding='SAME')                #          —— Weights0        
    xpp_ista  = tf.nn.relu(tf.nn.conv2d(xp_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    xppp_ista = tf.nn.conv2d(xpp_ista, Weights11, strides=[1, 1, 1, 1], padding='SAME')              #          —— Weights11   ——>>  F(p)    

    Frk_Fp =  x44_ista - xppp_ista


#   soft()
    x5_ista  = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(Frk_Fp) - soft_thr))                 #          ——>>  F(rk) - F(p)
    x55_ista = x5_ista + xppp_ista
    
 
#   F(x)^hat
    x6_ista  = tf.nn.relu(tf.nn.conv2d(x55_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))  
    x66_ista = tf.nn.conv2d(x6_ista, Weights22, strides=[1, 1, 1, 1], padding='SAME')                #          —— Weights22        ——>>  x_k   
    
    x7_ista  = tf.nn.conv2d(x66_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')                #          —— Weights3         ——>>  x_k     
    
    x7_ista  = x7_ista  + R_X                                                                      



    x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista,     Weights1, strides=[1, 1, 1, 1], padding='SAME'))    
    x4_ista_sym =            tf.nn.conv2d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')   
    
    x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym =            tf.nn.conv2d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')    
    
#   systematic
    x11_ista = x7_ista_sym - x3_ista                                                                        
    
    return [x_k, x11_ista, x7_ista]  








def add_con2d_weight_bias(w_shape, b_shape, order_no):
    
    with tf.variable_scope('weight_bias_scope', reuse=tf.AUTO_REUSE ):
        Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)     
        biases  = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
        return    [Weights, biases]






 
def compute_cost(Prediction,ListX ,P_input ,X_output, PhaseNumber):
    
    cost_sym = 0
    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.square(ListX[k]))
        

    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))  
    
    print('cost',type(cost))       
    return [cost, cost_sym]