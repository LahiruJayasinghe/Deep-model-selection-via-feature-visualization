from IPython.display import Image, display
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from skimage import io,transform

import inception
from cache import cache
from inception import transfer_values_cache
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
from get_csv_data import HandleData
from utils_laj import do_PCA
from utils_laj import plot_scatter
# Functions and classes for loading and using the Inception model.



inception.maybe_download()


model = inception.Inception()

def classify(image_path):
    # Display the image.
    display(Image(image_path))

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)


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

# def plot_scatter(values,cls,dim,label,label_x,label_y,label_z):
#
#     if dim==3:
#         colors = ['red', 'green', 'blue']
#         cmap = matplotlib.colors.ListedColormap(colors)
#
#         x = values[:, 0]
#         y = values[:, 1]
#         z = values[:, 2]
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111,projection='3d')
#         sc=ax.scatter(x,y,z,c=cls,cmap=cmap)
#         plt.title(label)
#         # plt.ylabel(label_x)
#         # plt.xlabel(label_y)
#         ax.set_xlabel(label_x)
#         ax.set_ylabel(label_y)
#         ax.set_zlabel(label_z)
#
#         tb = plt.table(cellText=[[x] for x in ['Dirty','Average','Clean']],
#                   loc='best',
#                   colWidths=[0.13],
#                   rowColours=cmap(np.array([0, 1, 2]))
#                   #cellColours=cmap(np.array([[0],[1],[2]]))
#
#                   )
#         tb.auto_set_font_size(False)
#         tb.set_fontsize(10)
#
#     else:
#         plt.figure()
#         # cmap = cm.Set1(np.linspace(0.0, 1.0, num_classes))
#         colors = ['red', 'green', 'blue']
#         classes = ['Dirty','Average','Clean']
#         cmap = matplotlib.colors.ListedColormap(colors)
#         # colors = cmap[cls]
#         x = values[:, 0]
#         y = values[:, 1]
#         plt.scatter(x,y ,c=cls,cmap=cmap,label=classes)
#         plt.title(label)
#         plt.xlabel(label_x)
#         plt.ylabel(label_y)
#
#         # cb = plt.colorbar()
#         # loc = np.arange(0, max(cls), max(cls) / float(len(colors)))
#         # cb.set_ticks(loc)
#         # cb.set_ticklabels(['dirty','average','clean'])
#         # unique_classes = list(set(classes))
#         tb = plt.table(cellText=[[x] for x in classes],
#                   loc='best',
#                   colWidths=[0.13],
#                   rowColours=cmap(np.array([0, 1, 2]))
#                   # cellColours=cmap(np.array([[0],[1],[2]]))
#
#                   )
#         tb.auto_set_font_size(False)
#         tb.set_fontsize(10)


def do_TSNE(transfer_values_train,cls_train,label='',pca_components=50):
    pca = PCA(n_components=pca_components)
    transfer_values_50d = pca.fit_transform(transfer_values_train)
    tsne = TSNE(n_components=3)
    transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
    print(label, tsne.kl_divergence_)
    plot_scatter(transfer_values_reduced, cls_train, 3, label+'TSNE 3D','tSNE x','tSNE y','tSNE z')
    plot_scatter(transfer_values_reduced, cls_train, 2, label+'TSNE 2D','tSNE x','tSNE y','tSNE z')

# def do_PCA(transfer_values_train,cls_train,label=''):
#     pca = PCA(n_components=3)
#     transfer_values_reduced = pca.fit_transform(transfer_values_train)
#     print(label,pca.explained_variance_ratio_,np.sum(pca.explained_variance_ratio_))
#     explain_var = pca.explained_variance_ratio_ * 100
#     label_x='PCA component 1\nExplained variance = '+str(round(explain_var[0],2))+'%'
#     label_y = 'PCA component 2\nExplained variance = ' + str(round(explain_var[1],2))+'%'
#     label_z = 'PCA component 3\nExplained variance = ' + str(round(explain_var[2],2))+'%'
#     plot_scatter(transfer_values_reduced, cls_train, 2, label,label_x,label_y,label_z)
#     plot_scatter(transfer_values_reduced, cls_train, 3, label,'PCA component 1\nExplained variance = '+str(round(explain_var[0],2))+'%','PCA component 2\nExplained variance = '+str(round(explain_var[1],2))+'%','PCA component 3\nExplained variance = '+str(round(explain_var[2],2))+'%')

# cifar10.maybe_download_and_extract()
# class_names = cifar10.load_class_names()
# images_train, cls_train, labels_train = cifar10.load_training_data()
# print(np.unique(cls_train))
# print(cifar10.data_path)

# images_train = []
path = r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\data\custom'
file_path_cache_train = os.path.join(path, 'inception_train.pkl')
file_path_cache_train_cls = os.path.join(path, 'inception_train_cls.pkl')
# imnames = glob.glob(r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\Urinal_images\*.JPG')
# imlist = (io.imread_collection(imnames))

# cat1 = (io.imread_collection(glob.glob(r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\Urinal_images\cat1\*.JPG')))
#
# cat2 = (io.imread_collection(glob.glob(r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\Urinal_images\cat2\*.JPG')))
#
# cat3 = (io.imread_collection(glob.glob(r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\Urinal_images\cat3\*.JPG')))


# for im in imlist:
#     images_train.append(im)
# images_train = np.array(images_train)
# print(images_train.shape)

# cls_train = []
# print("cat1")
# for im in cat1:
    # images_train.append(im)
    # cls_train.append(0)
# print("cat2")
# for im in cat2:
    # images_train.append(im)
    # cls_train.append(1)
# print("cat3")
# for im in cat3:
    # images_train.append(im)
    # cls_train.append(2)

# images_train = np.array(images_train)
# cls_train = np.array(cls_train)
cls_train = cache(file_path_cache_train_cls,fn=False)
images_train = cache(file_path_cache_train,fn=False)
# with open(file_path_cache_train_cls, mode='wb') as file:
#     pickle.dump(cls_train, file)

# print(images_train.shape,cls_train.shape)

data_set = HandleData(3)
data_set.total_data=227
data_set.data_set=images_train
data_set.label_set=cls_train

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_train,
                                              model=model)

print(transfer_values_train.shape,images_train.shape)

# plot_transfer_values(50,images_train,transfer_values_train)

Z, label_x, label_y, label_z=do_PCA(transfer_values_train,cls_train,'PCA analysis of extracted features\nDCNN = InceptionV3 , Layer = AvgPool')
# plot_scatter(Z[:,3:], cls_train, 3, 'other eigen values',label_x,label_y,label_z)
plt.show()

# pca = PCA(n_components=3)
# transfer_values_reduced = pca.fit_transform(transfer_values_train)
# plot_scatter(transfer_values_reduced,cls_train,2,'pca_2')
# plot_scatter(transfer_values_reduced,cls_train,3,'pca_3')

# do_TSNE(transfer_values_train,cls_train,'before training ')
# pca = PCA(n_components=50)
# transfer_values_50d = pca.fit_transform(transfer_values_train)
# tsne = TSNE(n_components=3)
# transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
# plot_scatter(transfer_values_reduced,cls_train,3,'TSNE_3d')
# plot_scatter(transfer_values_reduced,cls_train,2,'TSNE_2d')
# plt.show()

learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100
hidden_size = 1024
input_size = 2048
_SAVEFLAG = 0
_TRAINING = 0
_LOADDIR = './save/inception_v3/'

x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')
keep_prob = tf.placeholder(tf.float32)
# y_true_cls = tf.argmax(y_true, dimension=1)

weights = {
    'h1': tf.Variable(tf.random_normal([input_size, hidden_size])),
    'out': tf.Variable(tf.random_normal([hidden_size, 3]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_size])),
    'out': tf.Variable(tf.random_normal([3]))
}

def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1_drop, weights['out']) + biases['out']
    return out_layer,layer_1

logits,hidden_layer = neural_net(x)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    if _SAVEFLAG != 1 and _TRAINING != 1:
        saver_restore = tf.train.import_meta_graph(_LOADDIR+'DAEandDNN_save.meta')
        saver_restore.restore(sess, tf.train.latest_checkpoint(_LOADDIR))

    if _TRAINING == 1:
        for step in range(1, num_steps+1):
            batch_x, batch_y = data_set.next_batch(batch_size)
            batch_y = tf.one_hot(batch_y,3).eval()
            # print(batch_x.shape,batch_y.shape)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={x: batch_x, y_true: batch_y,keep_prob: 0.3})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,y_true: batch_y,keep_prob: 1})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

    print("Optimization Finished!")

    if _SAVEFLAG == 1:
        saver.save(sess, './data_aug/DAEandDNN_save')

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 1.0}))

    output,hidden = sess.run([logits,hidden_layer], feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 1.0})

    # plot_scatter(output,data_set.label_set,3,'after training output_layer')
    # plot_scatter(hidden, data_set.label_set, 3, 'after training hidden_layer')
    # do_TSNE(output, data_set.label_set, label='after training output_layer ',pca_components=3)
    do_PCA(output, data_set.label_set, label='After training\nOutput Layer PCA Analysis')

    # do_TSNE(hidden,data_set.label_set,label='after training hidden_layer ')
    Z, label_x, label_y, label_z=do_PCA(hidden, data_set.label_set, label='After training\nHidden Layer PCA Analysis')

    print(hidden.shape,output.shape)

    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    np.random.seed(42)
    k_cls = 3
    cluster = 'kmeans'
    if cluster == 'kmeans':
        # kmeans = KMeans(n_clusters=k_cls,init='k-means++',n_init=k_cls)
        kmeans = KMeans(n_clusters=k_cls)
        kmeans.fit(Z)
        labels = kmeans.labels_
    elif cluster == 'dbscan':
        db = DBSCAN(eps=1, min_samples=10).fit(Z)
        labels = db.labels_
    plot_scatter(Z, labels, 3, 'kmeans', label_x, label_y, label_z)
    plt.show()


