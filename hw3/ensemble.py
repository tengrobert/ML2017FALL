import numpy as np
import pandas as pd
# output1 = pd.read_csv('./output.csv')
# output1 = output1.as_matrix()
# output1 = output1[:,1]
# output2 = pd.read_csv('./output2.csv')
# output2 = output2.as_matrix()
# output2 = output2[:,1]
output3 = pd.read_csv('./output3.csv')
output3 = output3.as_matrix()
output3 = output3[:,1]
output4 = pd.read_csv('./output_2.csv')
output4 = output4.as_matrix()
output4 = output4[:,1]
output5 = pd.read_csv('./okok.csv')
output5 = output5.as_matrix()
output5 = output5[:,1]
arr = np.array([output3, output4, output5])
axis = 0
u, indices = np.unique(arr, return_inverse=True)
u = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                None, np.max(indices) + 1), axis=axis)]
print(u)

output = pd.DataFrame(u)
output.columns = ['label']
output.index = [list(range(0, 7178))]
output.index.name = 'id'
output.to_csv('./ensembleoutput4.csv')

zzz = pd.read_csv('./ensembleoutput3.csv')
zzz = zzz.as_matrix()
zzz = zzz[:,1]
print(np.sum(u != zzz))