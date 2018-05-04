from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
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

model = ResNet50(weights='imagenet')
model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

x=cache('toilet_images.pkl')
cls = cache('toilet_label.pkl')

# x=cache(r'D:\LAHIRU\Work\Toilet\transferlearning_ResNet_VGG16\save\vgg_16\testing\testing.pkl')
# cls = cache(r'D:\LAHIRU\Work\Toilet\transferlearning_ResNet_VGG16\save\vgg_16\testing\testing_cls.pkl')
# print(len(x))
# features = np.array(model.predict(x)).reshape(len(x),2048)
features = cache('./save/resnet_50/features.pkl')
print(features,features.shape)

Z,label_x,label_y,label_z=do_PCA(features,cls,'PCA analysis of extracted features\nDCNN = ResNet-50 , Layer = AvgPool')
plt.show()

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



from get_csv_data import HandleData
import tensorflow as tf

cls_train = cls
images_train = features
data_set = HandleData(3)
data_set.total_data=len(x)
data_set.data_set=images_train
data_set.label_set=cls_train

learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100
hidden_size = 1024
input_size = 2048
_SAVEFLAG = 0
_TRAINING = 0
_LOADDIR = './save/resnet_50/'

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

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 0.3}))

    output,hidden = sess.run([logits,hidden_layer], feed_dict={x: data_set.data_set,y_true: tf.one_hot(data_set.label_set, 3).eval(),keep_prob: 0.3})


    # do_PCA(output, data_set.label_set, label='After training\nOutput Layer PCA Analysis')

    Z, label_x, label_y, label_z=do_PCA(hidden, data_set.label_set, label='After training\nHidden Layer PCA Analysis')

    print(hidden.shape,output.shape)

    plt.show()