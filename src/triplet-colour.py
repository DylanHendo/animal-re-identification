import random
import numpy as np
import cv2
import os
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score

IMG_SIZE = 224
DATA = "tiger"

print("Loading in .npy data...")

if DATA == "tiger":
    X = np.load('/home/n10325701/data/X_tiger_c.npy')
    y = np.load('/home/n10325701/data/y_tiger_c.npy')

    X = X / 255.0

    y = [int(i) for i in y]
    y = np.asarray(y)

elif DATA == "chimp":
    X = np.load('/home/n10325701/data/X_chimp_c.npy')
    y = np.load('/home/n10325701/data/y_chimp_c.npy')
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 3)
    X = np.asarray(X).astype('float32')
    X = X / 255.0
else:
        # load whale data
    X = np.load('/home/n10325701/data/X_whale.npy')
    y = np.load('/home/n10325701/data/y_whale.npy')
    X = np.asarray(X).astype('float32')
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)


d = dict([(b,a+1) for a,b in enumerate(sorted(set(y)))])
mapped = [d[x] for x in y]
y = np.asarray(mapped)


# split into train/validation/test at 60/20/20
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))


datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,          # randomly rotate images by 10 degrees
    zoom_range=0.2,             # randomly zoom images by 20%
    horizontal_flip=True,       # randomly flip along horizontal flip
    rescale=1./255,             # scale imgs ?
)

def GetTripletData(imgs, labels, batch_size):
    """
    Get a triplet, where 1 item is the anchor, theres is 1 positive match and 1 negative match

    Args:
        imgs (list):
        labels (list):
        batch_size (int):

    Returns:
        [image_a, image_b, image_c]
    """

    image_a = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    image_b = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    image_c = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    
    for i in range(batch_size):
        idx1 = random.randint(0, len(imgs) - 1)
        idx2 = random.randint(0, len(imgs) - 1)
        idx3 = random.randint(0, len(imgs) - 1)

        while (labels[idx1] != labels[idx2]):
            idx2 = random.randint(0, len(imgs) - 1)            
                
        while (labels[idx1] == labels[idx3]):
            idx3 = random.randint(0, len(imgs) - 1)

        image_a[i, :, :, :] = imgs[idx1,:,:,:]
        image_b[i, :, :, :] = imgs[idx2,:,:,:]
        image_c[i, :, :, :] = imgs[idx3,:,:,:]

    return [image_a, image_b, image_c]


def TripleGenerator(imgs, labels, batch_size):
    while True:
        [image_a, image_b, image_c] = GetTripletData(imgs, labels, batch_size)

        # # pass image through generator with labels
        genX1 = datagen.flow(image_a, batch_size=batch_size, seed=42)
        genX2 = datagen.flow(image_b, batch_size=batch_size, seed=42)
        genX3 = datagen.flow(image_c, batch_size=batch_size, seed=42)

        # # get next batch, with augmentation
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()

        # yield [image_a, image_b, image_c], None
        yield [X1i, X2i, X3i], None


class TripletLossLayer(layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        
        anchor = K.l2_normalize(anchor, axis=1)
        positive = K.l2_normalize(positive, axis=1)
        negative = K.l2_normalize(negative, axis=1)

        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def return_inception_model():
    input_vector = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
    subnet = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=input_vector,
        input_shape=(IMG_SIZE,IMG_SIZE,3)
    )

    out = subnet.output
    out = layers.Flatten()(out)
    model = keras.Model(subnet.input, out, name="SubConvNet")
    return model

base_model = return_inception_model()

anchor_input = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
positive_input = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
negative_input = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))

embedding_anchor = base_model(anchor_input)
embedding_positive = base_model(positive_input)
embedding_negative = base_model(negative_input)

loss_layer = TripletLossLayer(alpha=1, name='triplet_loss_layer')([embedding_anchor, embedding_positive, embedding_negative])
triplet_network = keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)
triplet_network.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))

batch_size = 64
validation_size = len(x_val)  
num_epochs = 30
training_gen = TripleGenerator(x_train, y_train, batch_size)
x_validation = GetTripletData(x_val, y_val, validation_size)

triplet_network.fit(
    training_gen,
    steps_per_epoch = len(x_train) // batch_size, 
    batch_size=batch_size, 
    epochs=num_epochs, 
    shuffle=True, 
    validation_data=(x_validation, None), 
)


# L2 dist
def ComputeDistance(a,b):
    return np.sum(np.square(a-b))

def TripletCMC(network, dataset_test, idxcatalog=0, idxcandidate=1, nb_test_class=51):
    
    _,w,h,c = dataset_test.shape
    
    #generates embeddings for gallery set
    gallery_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        gallery_images[i,:,:,:] = dataset_test[i][idxcandidate,:,:]
    gallery_embeddings = network.predict(gallery_images)
    
    #generates embeddings for probe set
    probe_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        probe_images[i,:,:,:] = dataset_test[i][idxcatalog,:,:]
    probe_embeddings = network.predict(probe_images)
       
    ranks = np.zeros(nb_test_class)
        
    #for each gallery 
    for i in range(nb_test_class):
        predictionsdtype=[('class', int), ('dist', float)]
        predictions = np.zeros(nb_test_class, dtype=predictionsdtype)

        # compute distance between the each probe and gallery img
        for ref in range(nb_test_class):
            predictions[ref] = (ref, ComputeDistance(gallery_embeddings[i,:], probe_embeddings[ref,:])) 

        # sort predictions, ranked from the smallest distance from the candidate to the biggest
        sorted_predictions = np.sort(predictions, order='dist')
        rankedPredictions = sorted_predictions['class']
        
        if i in rankedPredictions:
            # which rank   
            firstOccurance = np.argmax(rankedPredictions == i)  
            #update ranks 
            for j in range(firstOccurance, nb_test_class):            
                ranks[j] +=1
    
    cmcScores = ranks / nb_test_class    
    return cmcScores

cmc = TripletCMC(base_model, x_test)
print(cmc[:5])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(cmc)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('CMC Curve')
plt.savefig(f"../results/{DATA}_triplet_CMC.png", bbox_inches='tight')




def compute_dist(a,b):
    return np.sum(np.square(a-b))

def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class
        
    Returns
        probs : array of shape (m,m) containing distances
    
    '''
    m = X.shape[0]
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    embeddings = network.predict(X) #Compute all embeddings for all pics with current network
    size_embedding = embeddings.shape[1]
    
    #For each pics of our dataset
    k = 0
    for i in range(m):
        #Against all other images
        for j in range(i+1,m):
            #compute the probability of being the right decision, 1 for right class, 0 for all other classes
            probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])
            if (Y[i]==Y[j]):
                y[k] = 1
            else:
                y[k] = 0
            k += 1
    return probs, y

def compute_metrics(probs, yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. 
                     thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    auc = roc_auc_score(yprobs, probs, multi_class='ovr')   # calculate AUC, multiclass
    fpr, tpr, thresholds = roc_curve(yprobs, probs)         # calculate roc curve
    return fpr, tpr, thresholds, auc

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1],idx-1
    else:
        return array[idx],idx
    
def draw_roc(fpr, tpr, thresholds):
    #find threshold
    targetfpr=1e-3
    _, idx = find_nearest(fpr,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]
    
    # plot no skill
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--') 
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}'.format(auc,targetfpr,recall))
    plt.xlabel('Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.savefig(f"../results/{DATA}_triplet_AUC.png", bbox_inches='tight')

probs, yprob = compute_probs(base_model, x_test, y_test)
fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
draw_roc(fpr, tpr, thresholds)

print(f"AUC: {auc}")


