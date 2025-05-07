# %% [markdown]
# ## Project 2 Spectrum Sensing with Deep Learning
# ECE 5625
#
# By Caleb Moore, Kevyn Kelso

# %% [markdown]
# # Introduction
#

# %% [markdown]
# ### Data Exploration
# images of the different signals
# Each class name [AM, FM, LTE, Noise, NR, Unknown] correspond to an index 0 - 5.
# Below are random samples of each type from the dataset.

# %%
import os
from constants import DATA_DIR
from IPython.display import Image, display

class_names = sorted(os.listdir(DATA_DIR))
for i in range(5):
    print(class_names[i])
    display(Image(filename=f'{i}.png'))


# %%
from constants import DATA_DIR
from networks import classification_network, deeper_cnn_network, residual_style_network
from training import evaluate_model_performance

model_funs = [classification_network, deeper_cnn_network, residual_style_network]
for fun in model_funs:
    model = fun()
    print(f"\n\nDeep learning model {model.friendly_name}")
    model.summary()
    evaluate_model_performance(model)

# %% [markdown]
# ### Model Architectures
# - classification_network
# Deep learning model classifier_v2
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 126, 126, 32)      896       
#                                                                  
#  max_pooling2d (MaxPooling2  (None, 63, 63, 32)        0         
#  D)                                                              
#                                                                  
#  conv2d_1 (Conv2D)           (None, 61, 61, 64)        18496     
#                                                                  
#  max_pooling2d_1 (MaxPoolin  (None, 30, 30, 64)        0         
#  g2D)                                                            
#                                                                  
#  conv2d_2 (Conv2D)           (None, 28, 28, 128)       73856     
#                                                                  
#  flatten (Flatten)           (None, 100352)            0         
#                                                                  
#  dense (Dense)               (None, 128)               12845184  
#                                                                  
#  dense_1 (Dense)             (None, 5)                 645       
#                                                                  
# =================================================================
# Total params: 12939077 (49.36 MB)
# Trainable params: 12939077 (49.36 MB)
# Non-trainable params: 0 (0.00 Byte)
# 
# - deeper_cnn_network
# Deep learning model deeper_cnn_v1
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_3 (Conv2D)           (None, 126, 126, 32)      896       
#                                                                  
#  batch_normalization (Batch  (None, 126, 126, 32)      128       
#  Normalization)                                                  
#                                                                  
#  max_pooling2d_2 (MaxPoolin  (None, 63, 63, 32)        0         
#  g2D)                                                            
#                                                                  
#  dropout (Dropout)           (None, 63, 63, 32)        0         
#                                                                  
#  conv2d_4 (Conv2D)           (None, 61, 61, 64)        18496     
#                                                                  
#  batch_normalization_1 (Bat  (None, 61, 61, 64)        256       
#  chNormalization)                                                
#                                                                  
#  max_pooling2d_3 (MaxPoolin  (None, 30, 30, 64)        0         
#  g2D)                                                            
#                                                                  
#  dropout_1 (Dropout)         (None, 30, 30, 64)        0         
#                                                                  
#  conv2d_5 (Conv2D)           (None, 28, 28, 128)       73856     
#                                                                  
#  batch_normalization_2 (Bat  (None, 28, 28, 128)       512       
#  chNormalization)                                                
#                                                                  
#  max_pooling2d_4 (MaxPoolin  (None, 14, 14, 128)       0         
#  g2D)                                                            
#                                                                  
#  dropout_2 (Dropout)         (None, 14, 14, 128)       0         
#                                                                  
#  conv2d_6 (Conv2D)           (None, 12, 12, 256)       295168    
#                                                                  
#  batch_normalization_3 (Bat  (None, 12, 12, 256)       1024      
#  chNormalization)                                                
#                                                                  
#  global_average_pooling2d (  (None, 256)               0         
#  GlobalAveragePooling2D)                                         
#                                                                  
#  dense_2 (Dense)             (None, 256)               65792     
#                                                                  
#  dropout_3 (Dropout)         (None, 256)               0         
#                                                                  
#  dense_3 (Dense)             (None, 5)                 1285      
#                                                                  
# =================================================================
# Total params: 457413 (1.74 MB)
# Trainable params: 456453 (1.74 MB)
# Non-trainable params: 960 (3.75 KB)

# - residual_style_network
# Deep learning model resnet_style_v1
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to                  
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 128, 128, 3)]        0         []                            
#                                                                                                   
#  conv2d_7 (Conv2D)           (None, 64, 64, 32)           4736      ['input_1[0][0]']             
#                                                                                                   
#  batch_normalization_4 (Bat  (None, 64, 64, 32)           128       ['conv2d_7[0][0]']            
#  chNormalization)                                                                                 
#                                                                                                   
#  activation (Activation)     (None, 64, 64, 32)           0         ['batch_normalization_4[0][0]'
#                                                                     ]                             
#                                                                                                   
#  max_pooling2d_5 (MaxPoolin  (None, 32, 32, 32)           0         ['activation[0][0]']          
#  g2D)                                                                                             
#                                                                                                   
#  conv2d_8 (Conv2D)           (None, 32, 32, 64)           18496     ['max_pooling2d_5[0][0]']     
#                                                                                                   
#  batch_normalization_5 (Bat  (None, 32, 32, 64)           256       ['conv2d_8[0][0]']            
#  chNormalization)                                                                                 
#                                                                                                   
#  activation_1 (Activation)   (None, 32, 32, 64)           0         ['batch_normalization_5[0][0]'
#                                                                     ]                             
#                                                                                                   
#  conv2d_9 (Conv2D)           (None, 32, 32, 32)           18464     ['activation_1[0][0]']        
#                                                                                                   
#  batch_normalization_6 (Bat  (None, 32, 32, 32)           128       ['conv2d_9[0][0]']            
#  chNormalization)                                                                                 
#                                                                                                   
#  add (Add)                   (None, 32, 32, 32)           0         ['batch_normalization_6[0][0]'
#                                                                     , 'max_pooling2d_5[0][0]']    
#                                                                                                   
#  conv2d_11 (Conv2D)          (None, 16, 16, 128)          36992     ['add[0][0]']                 
#                                                                                                   
#  batch_normalization_7 (Bat  (None, 16, 16, 128)          512       ['conv2d_11[0][0]']           
#  chNormalization)                                                                                 
#                                                                                                   
#  activation_2 (Activation)   (None, 16, 16, 128)          0         ['batch_normalization_7[0][0]'
#                                                                     ]                             
#                                                                                                   
#  conv2d_12 (Conv2D)          (None, 16, 16, 128)          147584    ['activation_2[0][0]']        
#                                                                                                   
#  batch_normalization_8 (Bat  (None, 16, 16, 128)          512       ['conv2d_12[0][0]']           
#  chNormalization)                                                                                 
#                                                                                                   
#  conv2d_10 (Conv2D)          (None, 16, 16, 128)          4224      ['add[0][0]']                 
#                                                                                                   
#  add_1 (Add)                 (None, 16, 16, 128)          0         ['batch_normalization_8[0][0]'
#                                                                     , 'conv2d_10[0][0]']          
#                                                                                                   
#  global_average_pooling2d_1  (None, 128)                  0         ['add_1[0][0]']               
#   (GlobalAveragePooling2D)                                                                        
#                                                                                                   
#  dense_4 (Dense)             (None, 256)                  33024     ['global_average_pooling2d_1[0
#                                                                     ][0]']                        
#                                                                                                   
#  dense_5 (Dense)             (None, 5)                    1285      ['dense_4[0][0]']             
#                                                                                                   
# ==================================================================================================
# Total params: 266341 (1.02 MB)
# Trainable params: 265573 (1.01 MB)
# Non-trainable params: 768 (3.00 KB)

# %% [markdown]
# ### Network Training
# Display pngs for accuracy per epoch, loss per epoch (explain loss function) categorical_crossentropy

# %% [markdown]
# ### Results
# TODO(kkelso): rerun with more than 50 samples
# Evaluation for classifier_v2
# Classification Report:
#               precision    recall  f1-score   support
#
#           AM       1.00      1.00      1.00        50
#           FM       1.00      1.00      1.00        50
#          LTE       1.00      1.00      1.00        50
#           NR       0.98      1.00      0.99        50
#        Noise       1.00      0.98      0.99        50
#
#     accuracy                           1.00       250
#    macro avg       1.00      1.00      1.00       250
# weighted avg       1.00      1.00      1.00       250
#
#
# Evaluation for deeper_cnn_v1
# Classification Report:
#               precision    recall  f1-score   support
#
#           AM       0.98      1.00      0.99        50
#           FM       1.00      1.00      1.00        50
#          LTE       1.00      1.00      1.00        50
#           NR       1.00      1.00      1.00        50
#        Noise       1.00      0.98      0.99        50
#
#     accuracy                           1.00       250
#    macro avg       1.00      1.00      1.00       250
# weighted avg       1.00      1.00      1.00       250
#
#
# Evaluation for resnet_style_v1
# Classification Report:
#               precision    recall  f1-score   support
#
#           AM       1.00      1.00      1.00        50
#           FM       1.00      1.00      1.00        50
#          LTE       1.00      1.00      1.00        50
#           NR       1.00      1.00      1.00        50
#        Noise       1.00      1.00      1.00        50
#
#     accuracy                           1.00       250
#    macro avg       1.00      1.00      1.00       250
# weighted avg       1.00      1.00      1.00       250
#
# Display the confusion matrixes
