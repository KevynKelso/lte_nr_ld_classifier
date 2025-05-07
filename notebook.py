# %% [markdown]
# ## Project 2 Spectrum Sensing with Deep Learning
# ### ECE 5625
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
    display(Image(filename=f'final_plots/{i}.png'))


# %% [markdown]
# ### Model Architectures
# #### Deep learning model: classifier_v2
# | Layer (Type)               | Output Shape        | Param #   | Connected To            |
# |----------------------------|---------------------|-----------|-------------------------|
# | conv2d (Conv2D)            | (None, 126, 126, 32)| 896       |                         |
# | max_pooling2d (MaxPooling2D)| (None, 63, 63, 32)  | 0         |                         |
# | conv2d_1 (Conv2D)          | (None, 61, 61, 64)  | 18,496    |                         |
# | max_pooling2d_1 (MaxPooling2D)| (None, 30, 30, 64)| 0         |                         |
# | conv2d_2 (Conv2D)          | (None, 28, 28, 128) | 73,856    |                         |
# | flatten (Flatten)          | (None, 100352)      | 0         |                         |
# | dense (Dense)              | (None, 128)         | 12,845,184|                         |
# | dense_1 (Dense)            | (None, 5)           | 645       |                         |
# 
# **Total params**: 12,939,077 (49.36 MB)  
# **Trainable params**: 12,939,077 (49.36 MB)  
# **Non-trainable params**: 0 (0.00 Byte)
# 
# #### Deep learning model deeper_cnn_v1
# | Layer (Type)               | Output Shape        | Param #   | Connected To            |
# |----------------------------|---------------------|-----------|-------------------------|
# | conv2d_3 (Conv2D)          | (None, 126, 126, 32)| 896       |                         |
# | batch_normalization        | (None, 126, 126, 32)| 128       |                         |
# | max_pooling2d_2 (MaxPooling2D)| (None, 63, 63, 32)| 0         |                         |
# | dropout                    | (None, 63, 63, 32) | 0         |                         |
# | conv2d_4 (Conv2D)          | (None, 61, 61, 64) | 18,496    |                         |
# | batch_normalization_1      | (None, 61, 61, 64) | 256       |                         |
# | max_pooling2d_3 (MaxPooling2D)| (None, 30, 30, 64)| 0         |                         |
# | dropout_1                  | (None, 30, 30, 64) | 0         |                         |
# | conv2d_5 (Conv2D)          | (None, 28, 28, 128)| 73,856    |                         |
# | batch_normalization_2      | (None, 28, 28, 128)| 512       |                         |
# | max_pooling2d_4 (MaxPooling2D)| (None, 14, 14, 128)| 0         |                         |
# | dropout_2                  | (None, 14, 14, 128)| 0         |                         |
# | conv2d_6 (Conv2D)          | (None, 12, 12, 256)| 295,168   |                         |
# | batch_normalization_3      | (None, 12, 12, 256)| 1,024     |                         |
# | global_average_pooling2d   | (None, 256)         | 0         |                         |
# | dense_2 (Dense)            | (None, 256)         | 65,792    |                         |
# | dropout_3                  | (None, 256)         | 0         |                         |
# | dense_3 (Dense)            | (None, 5)           | 1,285     |                         |
# 
# **Total params**: 457,413 (1.74 MB)  
# **Trainable params**: 456,453 (1.74 MB)  
# **Non-trainable params**: 960 (3.75 KB)

# #### Deep learning model resnet_style_v1
# | Layer (Type)               | Output Shape        | Param #   | Connected To            |
# |----------------------------|---------------------|-----------|-------------------------|
# | input_1 (InputLayer)       | (None, 128, 128, 3)| 0         |                         |
# | conv2d_7 (Conv2D)          | (None, 64, 64, 32) | 4,736     | input_1[0][0]           |
# | batch_normalization_4      | (None, 64, 64, 32) | 128       | conv2d_7[0][0]          |
# | activation                 | (None, 64, 64, 32) | 0         | batch_normalization_4[0][0]|
# | max_pooling2d_5 (MaxPooling2D)| (None, 32, 32, 32)| 0         | activation[0][0]        |
# | conv2d_8 (Conv2D)          | (None, 32, 32, 64) | 18,496    | max_pooling2d_5[0][0]   |
# | batch_normalization_5      | (None, 32, 32, 64) | 256       | conv2d_8[0][0]          |
# | activation_1               | (None, 32, 32, 64) | 0         | batch_normalization_5[0][0]|
# | conv2d_9 (Conv2D)          | (None, 32, 32, 32) | 18,464    | activation_1[0][0]      |
# | batch_normalization_6      | (None, 32, 32, 32) | 128       | conv2d_9[0][0]          |
# | add                        | (None, 32, 32, 32) | 0         | batch_normalization_6[0][0], max_pooling2d_5[0][0]|
# | conv2d_11 (Conv2D)         | (None, 16, 16, 128)| 36,992    | add[0][0]               |
# | batch_normalization_7      | (None, 16, 16, 128)| 512       | conv2d_11[0][0]         |
# | activation_2               | (None, 16, 16, 128)| 0         | batch_normalization_7[0][0]|
# | conv2d_12 (Conv2D)         | (None, 16, 16, 128)| 147,584   | activation_2[0][0]      |
# | batch_normalization_8      | (None, 16, 16, 128)| 512       | conv2d_12[0][0]         |
# | conv2d_10 (Conv2D)         | (None, 16, 16, 128)| 4,224     | add[0][0]               |
# | add_1                      | (None, 16, 16, 128)| 0         | batch_normalization_8[0][0], conv2d_10[0][0]|
# | global_average_pooling2d_1 | (None, 128)        | 0         | add_1[0][0]             |
# | dense_4 (Dense)            | (None, 256)        | 33,024    | global_average_pooling2d_1[0][0]|
# | dense_5 (Dense)            | (None, 5)          | 1,285     | dense_4[0][0]           |
# 
# **Total params**: 266,341 (1.02 MB)  
# **Trainable params**: 265,573 (1.01 MB)  
# **Non-trainable params**: 768 (3.00 KB)

# %% [markdown]
# ### Network Training
# Display pngs for accuracy per epoch, loss per epoch (explain loss function) categorical_crossentropy

# %% [markdown]
# ### Results
# #### Evaluation for classifier_v2
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.99      | 1.00   | 1.00     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 0.99      | 0.99   | 0.99     | 500     |
# | NR      | 1.00      | 0.99   | 0.99     | 500     |
# | Noise   | 1.00      | 1.00   | 1.00     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **1.00** | **2500** |
# | **Macro Avg** | 1.00   | 1.00   | **1.00** | **2500** |
# | **Weighted Avg** | 1.00 | 1.00 | **1.00** | **2500** |

# #### Evaluation for deeper_cnn_v1
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.98      | 1.00   | 0.99     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 1.00      | 1.00   | 1.00     | 500     |
# | NR      | 1.00      | 1.00   | 1.00     | 500     |
# | Noise   | 1.00      | 0.97   | 0.99     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **0.99** | **2500** |
# | **Macro Avg** | 0.99   | 0.99   | **0.99** | **2500** |
# | **Weighted Avg** | 0.99 | 0.99 | **0.99** | **2500** |

# #### Evaluation for resnet_style_v1
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.99      | 1.00   | 1.00     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 1.00      | 1.00   | 1.00     | 500     |
# | NR      | 0.99      | 1.00   | 1.00     | 500     |
# | Noise   | 1.00      | 0.99   | 1.00     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **1.00** | **2500** |
# | **Macro Avg** | 1.00   | 1.00   | **1.00** | **2500** |
# | **Weighted Avg** | 1.00 | 1.00 | **1.00** | **2500** |
#

# %% [markdown]
# ### Confusion matrix for classifier_v2 network
# %%
display(Image(filename=f'final_plots/classifier_v2-confusion-matrix.png'))

# %% [markdown]
# ### Confusion matrix for deeper_cnn_v1 network
# %%
display(Image(filename=f'final_plots/deeper_cnn_v1-confusion-matrix.png'))

# %% [markdown]
# ### Confusion matrix for resnet_style_v1 network
# %%
display(Image(filename=f'final_plots/resnet_style_v1-confusion-matrix.png'))
