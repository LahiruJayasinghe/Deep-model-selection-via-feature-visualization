import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.manifold import TSNE
import glob
from PIL import Image
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
import matplotlib.cm as cm
import matplotlib
from numpy import linalg as LA

def cache(cache_path,obj=0):
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)
        print("- Data loaded from cache-file: " + cache_path)
    else:
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)

    return obj

def movingavg(data,window): #[n_samples, n_features]
    data = np.transpose(data)
    if data.ndim > 1 :
        tmp = []
        for i in range(data.shape[0]):
            ma = movingavg(np.squeeze(data[i]), window)
            tmp.append(ma)
        smas = np.array(tmp)
    else :
        w = np.repeat(1.0,window)/window
        smas = np.convolve(data,w,'valid')
    smas = np.transpose(smas)
    return smas #[n_samples, n_features]

def imageresize(imlist,resize_shape=(256, 256, 3)):
    resize_im = []
    for i in range(len(imlist)):
        m = transform.resize(imlist[i], resize_shape)
        resize_im.append(m)

    return resize_im

def imagenomalization(image,nMax,nMin):
    imin = np.min(image)
    imax = np.max(image)
    if len(image.shape) == 3:
        n_dim = 3
        for i in range(0,n_dim):
            image[:,:,i] = (image[:,:,i]-imin) * ((nMax-nMin)/(imax-imin)) + nMin
    else:
        image = (image-imin) * ((nMax-nMin)/(imax-imin)) + nMin

    return image

def showimg(image,title=''):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    # plt.draw()

def plot_transfer_values(i,imlist,transfer_values):
    print("Input image:")

    # Plot the i'th image from the test-set.
    plt.figure()
    plt.imshow(imlist[i].reshape((32, 64)), interpolation='nearest')
    # plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.show()

def plot_scatter(values,cls,dim,label,label_x,label_y,label_z):

    if dim==3:
        colors = ['red', 'green', 'blue']
        cmap = matplotlib.colors.ListedColormap(colors)

        x = values[:, 0]
        y = values[:, 1]
        z = values[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        sc=ax.scatter(x,y,z,c=cls,cmap=cmap)
        plt.title(label)
        # plt.ylabel(label_x)
        # plt.xlabel(label_y)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_zlabel(label_z)

        tb = plt.table(cellText=[[x] for x in ['Dirty','Average','Clean']],
                  loc='best',
                  colWidths=[0.13],
                  rowColours=cmap(np.array([0, 1, 2]))
                  #cellColours=cmap(np.array([[0],[1],[2]]))

                  )
        tb.auto_set_font_size(False)
        tb.set_fontsize(10)

    else:
        plt.figure()
        # cmap = cm.Set1(np.linspace(0.0, 1.0, num_classes))
        colors = ['red', 'green', 'blue']
        classes = ['Dirty','Average','Clean']
        cmap = matplotlib.colors.ListedColormap(colors)
        # colors = cmap[cls]
        x = values[:, 0]
        y = values[:, 1]
        plt.scatter(x,y ,c=cls,cmap=cmap,label=classes)
        plt.title(label)
        plt.xlabel(label_x)
        plt.ylabel(label_y)

        # cb = plt.colorbar()
        # loc = np.arange(0, max(cls), max(cls) / float(len(colors)))
        # cb.set_ticks(loc)
        # cb.set_ticklabels(['dirty','average','clean'])
        # unique_classes = list(set(classes))
        tb = plt.table(cellText=[[x] for x in classes],
                  loc='best',
                  colWidths=[0.13],
                  rowColours=cmap(np.array([0, 1, 2]))
                  # cellColours=cmap(np.array([[0],[1],[2]]))

                  )
        tb.auto_set_font_size(False)
        tb.set_fontsize(10)


def do_TSNE(transfer_values_train,cls_train,label='',pca_components=50):
    pca = PCA(n_components=pca_components)
    transfer_values_50d = pca.fit_transform(transfer_values_train)
    tsne = TSNE(n_components=3)
    transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
    print(label, tsne.kl_divergence_)
    plot_scatter(transfer_values_reduced, cls_train, 3, label+'TSNE 3D','tSNE x','tSNE y','tSNE z')
    plot_scatter(transfer_values_reduced, cls_train, 2, label+'TSNE 2D','tSNE x','tSNE y','tSNE z')

def do_PCA(transfer_values_train,cls_train,label='',plot=True,pca_components=3):
    pca = PCA(n_components=pca_components)
    transfer_values_reduced = pca.fit_transform(transfer_values_train)
    print(label,pca.explained_variance_ratio_,np.sum(pca.explained_variance_ratio_))
    explain_var = pca.explained_variance_ratio_ * 100
    label_x='PCA component 1\nExplained variance = '+str(round(explain_var[0],2))+'%'
    label_y = 'PCA component 2\nExplained variance = ' + str(round(explain_var[1],2))+'%'
    label_z = 'PCA component 3\nExplained variance = ' + str(round(explain_var[2],2))+'%'
    if plot==True:
        plot_scatter(transfer_values_reduced, cls_train, 2, label,label_x,label_y,label_z)
        plot_scatter(transfer_values_reduced, cls_train, 3, label,'PCA component 1\nExplained variance = '+str(round(explain_var[0],2))+'%','PCA component 2\nExplained variance = '+str(round(explain_var[1],2))+'%','PCA component 3\nExplained variance = '+str(round(explain_var[2],2))+'%')
    return transfer_values_reduced,label_x,label_y,label_z