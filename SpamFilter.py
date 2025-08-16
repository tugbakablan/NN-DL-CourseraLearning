# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:17:25 2025

@author: TUĞBA KABLAN
"""

%matplotlib inline
import numpy as np
from collections import Counter

##Load the dataset (if in the same directory as the notebook)
sms_data = np.loadtxt("C:\Users\TUĞBA KABLAN\Desktop\ML-Spam Clasifier\02-2-SpamClassifier/SMSSpamCollection_cleaned.csv", delimiter="\t", skiprows=1, dtype=str)

## create test data set for checkpointing
checkpoint_data = np.array([['spam', 'dear researcher submit manuscript money'], 
          ['ham','dear friend meet beer'],
          ['ham', 'dear friend meet you']], dtype=str) 