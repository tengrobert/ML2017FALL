import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

model = load_model('mfta-05-1.1459.h5')

user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user emb shape: ',user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie emb shape: ',movie_emb.shape)

def draw(x, y):
    y = np.array(y)
    x = np.array(x, dtype=np.float64)
    vis_data = TSNE(n_components=2).fit_transform(x)
    print('plot')
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    #plt.colorbar(sc)
    plt.show()

zzz = np.random.rand(3706)
draw(movie_emb, zzz)
