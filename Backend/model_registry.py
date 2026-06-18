import pandas as pd
import numpy as np

def identify_task(df,target_column):
   if df[target_column].dtype in [np.float64, np.int64]:
      return "regression"
   else:
      return "classification"
