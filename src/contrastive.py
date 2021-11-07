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

# for consistency
IMG_SIZE = 250
DATA = "whale"

print("Loading in .npy data...")

if DATA == "tiger":
    # load tiger data
    X = np.load('../data/X_tiger.npy')
    y = np.load('../data/y_tiger.npy')
    y = [int(i) for i in y]
    y = np.asarray(y)
elif DATA == "chimp":
    # load chimp data
    num_epochs=50
    X_t = np.load('../data/X_tai.npy')
    y_t = np.load('../data/y_tai.npy')
    X_t = X_t.reshape(X_t.shape[0], IMG_SIZE, IMG_SIZE, 1)

    X_z = np.load('../data/X_zoo.npy')
    y_z = np.load('../data/y_zoo.npy')

    X = np.concatenate((X_t, X_z))
    y = np.concatenate((y_t, y_z))

    X = np.asarray(X).astype('float32')
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
else:
    # load whale data
    num_epochs=30
    X = np.load('/home/n10325701/data/X_whale.npy')
    y = np.load('/home/n10325701/data/y_whale.npy')
    X = np.asarray(X).astype('float32')
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)

print(DATA + " data loaded successfully!\n")


# map unique chimp names to unique ints, for plotting purposes
y_dict = dict([(b,a+1) for a,b in enumerate(sorted(set(y)))])
mapped = [y_dict[x] for x in y]
y = np.asarray(mapped)

# split into train/validation/test at 60/20/20
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.2,             # randomly zoom images by 20%
    horizontal_flip=True,       # randomly flip along horizontal flip
)

# generate stream of data
def GetSiameseData(imgs, labels, batch_size):
    image_a = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    image_b = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    label = np.zeros(batch_size)
    
    for i in range(batch_size):
        
        if (i % 2 == 0):
            idx1 = random.randint(0, len(imgs) - 1)
            idx2 = random.randint(0, len(imgs) - 1)
            l = 1
            while (labels[idx1] != labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)            
                
        else:
            idx1 = random.randint(0, len(imgs) - 1)
            idx2 = random.randint(0, len(imgs) - 1)
            l = 0
            while (labels[idx1] == labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)

        image_a[i, :, :, :] = imgs[idx1,:,:,:]
        image_b[i, :, :, :] = imgs[idx2,:,:,:]
        label[i] = l

    return [image_a, image_b], label


def PairGenerator(imgs, labels, batch_size):
    while True:
        [image_a, image_b], label = GetSiameseData(imgs, labels, batch_size)

        # pass image through generator with labels
        genX1 = datagen.flow(image_a, label,  batch_size=batch_size, seed=42)
        genX2 = datagen.flow(image_a, image_b, batch_size=batch_size, seed=42)

        # get next batch, with augmentation
        X1i = genX1.next()
        X2i = genX2.next()

        # yield [image_a, image_b], label
        yield [X1i[0], X2i[1] ], X1i[1]    

# network
def conv_block(inputs, filters, spatial_dropout = 0.0, max_pool = True):
    
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (spatial_dropout > 0.0):
        x = layers.SpatialDropout2D(spatial_dropout)(x)
    if (max_pool == True):
        x = layers.MaxPool2D(pool_size=(2, 2))(x)
    
    return x

def fc_block(inputs, size, dropout):
    x = layers.Dense(size, activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if (dropout > 0.0):
        x = layers.Dropout(dropout)(x)
    
    return x

def vgg_net(inputs, filters, fc, spatial_dropout = 0.0, dropout = 0.0):
    
    x = inputs
    for idx,i in enumerate(filters):
        x = conv_block(x, i, spatial_dropout, not (idx==len(filters) - 1))
    
    x = layers.Flatten()(x)
    
    for i in fc:
        x = fc_block(x, i, dropout)
        
    return x


# embedding
embedding_size = 32
dummy_input = keras.Input((IMG_SIZE, IMG_SIZE, 1))
base_network = vgg_net(dummy_input, [8, 16, 32], [256], 0.2, 0)
embedding_layer = layers.Dense(embedding_size, activation=None)(base_network)
base_network = keras.Model(dummy_input, embedding_layer, name='SiameseBranch')

input_a = keras.Input((IMG_SIZE, IMG_SIZE, 1), name='InputA')
input_b = keras.Input((IMG_SIZE, IMG_SIZE, 1), name='InputB')

embedding_a = base_network(input_a)
embedding_b = base_network(input_b)


# contrastive loss functions
def euclidean_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

print("Compiling model ...")

# compile model
distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([embedding_a, embedding_b])
siamese_network = keras.Model([input_a, input_b], distance)
siamese_network.compile(loss=contrastive_loss, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

print("Training model...")

# train
batch_size = 64
training_gen = PairGenerator(x_train, y_train, batch_size)
siamese_test_x, siamese_test_y = GetSiameseData(x_val, y_val, len(x_val))

siamese_network.fit(
    training_gen, 
    steps_per_epoch = len(x_train) // batch_size, 
    epochs=num_epochs, 
    validation_data = (siamese_test_x, siamese_test_y)
)


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

    # plot ROC 
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}'.format(auc,targetfpr,recall))
    plt.xlabel('Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.savefig(f"../results/{DATA}_contrastive_AUC.png", bbox_inches='tight')


probs, yprob = compute_probs(base_network, x_test, y_test)
fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
draw_roc(fpr, tpr, thresholds)

print(f"AUC: {auc}")





# L2 dist
def ComputeDistance(a,b):
    return np.sum(np.square(a-b))

def CMC(network, dataset_test, idxcatalog=0, idxcandidate=1, nb_test_class=51):

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

cmc = CMC(base_network, x_test)
print(cmc[:5])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(cmc)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('CMC Curve')
plt.savefig(f"../results/{DATA}_contrastive_CMC.png", bbox_inches='tight')


