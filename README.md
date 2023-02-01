# Introduction

Modern machine learning suffers from catastrophic forgetting when learning new classes incrementally. The performance dramatically degrades due to the missing data of old classes. Incremental learning methods have been proposed to retain the knowledge acquired from the old classes, by using knowledge distilling and keeping a few exemplars
from the old classes. However, these methods struggle to scale up to a large number of classes. 
We believe this is because of the combination of two factors:   
	
	(a) the data imbalance between the old and new classes, and     
	(b) the increasing number of visually similar classes.  

Distinguishing between an increasing number of visually similar classes is particularly challenging, when the training data is unbalanced. We propose a simple and effective method to address this data imbalance issue. We found that the last fully connected layer has a strong bias towards the new classes, and this bias can be corrected by a linear model.

![](/images/incremental_learning.png)


A pytorch implementation of "Large Scale Incremental Learning" from https://arxiv.org/abs/1905.13260

# Dataset
Download Food101 dataset from https://www.kaggle.com/datasets/dansbecker/food-101

Put training and testing images into train, test folders

### Installation 

1. Install `Python 3.10` on your Local Machine 
2. Execute `git clone https://github.com/Abhinav1004/Large-Scale-Incremental-Learning.git` to clone the repository
3. Create Python Virtual Environment in root folder by opening terminal and executing
    ```
      * pip install virtualenv
      * virtualenv distracted_env
      * source distracted_env/bin/activate
     ```
4. Install required Python Libraries by `pip install -r requirements.txt`



# Train
```
python main.py

```

# Result

|    |  20  |  40  |  60  |  80  |  100  |
| ---- | ---- | ---- | ---- | ---- | ---- |
|  Paper  | 85.20 | 74.59 | 66.76 | 60.14 | 55.55 |
|  Implementation  | 83.80| 68.75| 63.50| 58.25| 54.93 |



# Alpha & Beta

### Adam (Bias correction layer)
|     |  20  |  40  |  60  |  80  |  100  |
| --- | ---- | ---- | ---- | ---- | ---- |
| Alpha | 1.0 | 0.788 | 0.718 | 0.700 | 0.696 |
| Beta | 0.0 | -0.289 | -0.310 | -0.325 | -0.327 |

### SGD (Bias correction layer)
|     |  20  |  40  |  60  |  80  |  100  |
| --- | ---- | ---- | ---- | ---- | ---- |
| Alpha | 1.0 | 1.006 | 1.017 | 0.976 | 0.983 |
| Beta | 0.0 | -2.809 | -3.496 | -3.447 | -3.683 |

Different Optimizers make difference in alpha and beta.

