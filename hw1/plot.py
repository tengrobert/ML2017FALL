from matplotlib import pyplot as plt
x = [0.1, 0.01, 0.001, 0.0001]
y_private = [6.09439, 6.20416, 6.21221, 6.21221]
y_public = [7.64331, 6.92760, 6.92880, 6.92579]
plt.plot(x, y_private, label='private')
plt.plot(x, y_public, label='public')
plt.xlim(0.000001,0.1)
plt.xlabel('lambda')
plt.ylabel('score')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()