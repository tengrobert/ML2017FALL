import pandas as pd
import numpy as np

output1 = pd.read_csv("./semitry.csv")
output1 = output1.as_matrix()
output2 = pd.read_csv('./output_11222333_8032.csv')
output2 = output2.as_matrix()
print(np.sum(output1 != output2) / 200000)
