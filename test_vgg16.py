from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import os
from utils_laj import cache
from utils_laj import do_PCA
from utils_laj import plot_scatter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

model = VGG16(weights='imagenet')
model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

# img_path = r'D:\LAHIRU\Work\Toilet\transferlearning_ResNet_VGG16\data\cats\cat.26.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# features = np.array(model.predict(x))
# print(features,features.shape)


# data_path = r'D:\LAHIRU\Work\Toilet\inceptionV3_custom\Urinal_images\New folder'
# data_path = r'D:\LAHIRU\Work\Toilet\InceptionV3_model_toilet_confusion_metrix\test\New folder'
# data_dir_list = os.listdir(data_path)
# print(data_dir_list)
# img_data_list = []
# data_per_cls = []
# for dataset in data_dir_list:
#     img_list = os.listdir(data_path + '/' + dataset)
#     data_per_cls.append(len(img_list))
#     print('Loaded the images of dataset-' + '{}\n'.format(dataset))
#     for img in img_list:
#         img_path = data_path + '/' + dataset + '/' + img
#         img = image.load_img(img_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         # print('img_to_array', x.shape)
#         x = np.expand_dims(x, axis=0)
#         # print('expand_dim',x.shape)
#         x = preprocess_input(x)
#         #		x = x/255
#         # print('Input image shape:', x.shape)
#         img_data_list.append(x)
#
# img_data = np.array(img_data_list)
# # img_data = img_data.astype('float32')
# print(img_data.shape)
# img_data = np.rollaxis(img_data, 1, 0)
# print(img_data.shape)
# img_data = img_data[0]
# print(img_data.shape)
#
# # Define the number of classes
# num_classes = len(data_dir_list)
# num_of_samples = img_data.shape[0]
# labels = np.ones((num_of_samples,), dtype='int64')
# print(data_per_cls)
# j=0
# k=0
# for i in data_per_cls:
#     labels[j:i+j] = k
#     print(j,i+j)
#     j = j+i
#     k = k+1

x=cache('toilet_images.pkl')
# x=cache(r'D:\LAHIRU\Work\Toilet\transferlearning_ResNet_VGG16\save\vgg_16\testing\testing.pkl',img_data)
# cls = cache(r'D:\LAHIRU\Work\Toilet\transferlearning_ResNet_VGG16\save\vgg_16\testing\testing_cls.pkl',labels)
cls = cache('toilet_label.pkl')

features = np.array(model.predict(x))

features = cache('./save/vgg_16/features-fc2.pkl',features)
# features = cache('./save/vgg_16/testing/features.pkl')
print(features,features.shape)
Z,label_x,label_y,label_z=do_PCA(features,cls,'PCA analysis of extracted features\nDCNN = VGG-16 , Layer = FC-2')

# np.random.seed(42)
# k_cls = 3
# cluster = 'kmeans'
# if cluster == 'kmeans':
#     # kmeans = KMeans(n_clusters=k_cls,init='k-means++',n_init=k_cls)
#     kmeans = KMeans(n_clusters=k_cls)
#     kmeans.fit(Z)
#     labels = kmeans.labels_
# elif cluster == 'dbscan':
#     db = DBSCAN(eps=1, min_samples=10).fit(Z)
#     labels = db.labels_

# plot_scatter(Z, labels, 3, 'Before training\nTransfer values PCA analysis',label_x,label_y,label_z)

plt.show()

from get_csv_data import HandleData
import tensorflow as tf

cls_train = cls
images_train = features
data_set = HandleData(3)
data_set.total_data=227
data_set.data_set=images_train
data_set.label_set=cls_train

learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100
hidden_size = 1024
input_size = 4096
_SAVEFLAG = 0
_TRAINING = 0
_LOADDIR = './save/vgg_16/'

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
        saver.save(sess, _LOADDIR+'DAEandDNN_save')

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 1}))

    output,hidden = sess.run([logits,hidden_layer], feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 1})


    # do_PCA(output, data_set.label_set, label='After training\nOutput Layer PCA Analysis')

    # Z, label_x, label_y, label_z=do_PCA(hidden, data_set.label_set, label='After training\nHidden Layer PCA Analysis')

    # np.random.seed(42)
    # k_cls = 3
    # cluster = 'kmeans'
    # if cluster == 'kmeans':
    #     # kmeans = KMeans(n_clusters=k_cls,init='k-means++',n_init=k_cls)
    #     kmeans = KMeans(n_clusters=k_cls)
    #     kmeans.fit(Z)
    #     labels = kmeans.labels_
    # elif cluster == 'dbscan':
    #     db = DBSCAN(eps=1, min_samples=10).fit(Z)
    #     labels = db.labels_
    #
    # plot_scatter(Z, labels, 3, 'kmeans', label_x, label_y, label_z)

    print(hidden.shape,output.shape)

    plt.show()