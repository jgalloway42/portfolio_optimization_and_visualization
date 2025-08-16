# Generic Jupyter Notebook Imports and Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
import sys
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (15,5)
pd.set_option('display.max_columns',50)
np.set_printoptions(precision=3)

# Path definitions
cwd = os.getcwd()
root_dir = os.path.abspath(os.path.join(cwd, os.pardir))
raw_data = os.path.join(root_dir,'data','raw')
processed_data = os.path.join(root_dir,'data','processed')
interim_data = os.path.join(root_dir,'data','interim')
external_data = os.path.join(root_dir,'data','external')
models_path = os.path.join(root_dir,'models')
source_path = os.path.join(root_dir,'src')
figures_path = os.path.join(root_dir,'reports','figures')

# Make src imports easier
sys.path.append(source_path)