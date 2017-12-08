import numpy as np
import pandas as pd

output1 = pd.read_csv('./semitry.csv')
output1 = output1.as_matrix()
output1 = output1[:,1]
output2 = pd.read_csv('./output_11300235_8030.csv')
output2 = output2.as_matrix()
output2 = output2[:,1]
output3 = pd.read_csv('./output_11300242_8028.csv')
output3 = output3.as_matrix()
output3 = output3[:,1]
# output4 = pd.read_csv('./output4.csv')
# output4 = output4.as_matrix()
# output4 = output4[:,1]
# output5 = pd.read_csv('./output5.csv')
# output5 = output5.as_matrix()
# output5 = output5[:,1]
arr = np.array([output1, output2, output3])
axis = 0
u, indices = np.unique(arr, return_inverse=True)
u = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),
                                None, np.max(indices) + 1), axis=axis)]
output = pd.DataFrame(u)
output.columns = ['label']
output.index = list(range(0, 200000))
output.index.name = 'id'
output.to_csv('./passs.csv', encoding='utf-8', index=True, index_label='id')
