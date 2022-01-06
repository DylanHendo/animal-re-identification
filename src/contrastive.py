import random
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score

IMG_SIZE = 224

DATA = "tiger"

print("Loading in .npy data...")

# load in relevant data, depending on dataset used
if DATA == "tiger":
    X = np.load('../data/X_tiger.npy')
    y = np.load('../data/y_tiger.npy')
elif DATA == "chimp":
    X = np.load('../data/X_chimp.npy')
    y = np.load('../data/y_chimp.npy')
else:
    X = np.load('../data/X_whale.npy')
    y = np.load('../data/y_whale.npy')

# normalise img data
X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 3) 
X = np.asarray(X).astype('float32')
X = X / 255.0

print(DATA + " data loaded successfully!\n")

# set for Y
d = dict([(b,a+1) for a,b in enumerate(sorted(set(y)))])
mapped = [d[x] for x in y]
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
    rescale=1./255,             # scale imgs ?
) 

# DATA functions
def GetSiameseData(imgs, labels, batch_size):
    """
    Get a number of pairs, and for each pair get a label indicating if the two imgs are of same type, 
    or different types. 
    It will always generated balanced data ( number of same pairs = number of different pairs).

    Args:
        imgs (list):
        labels (list):
        batch_size (int):

    Returns:
        [image_a, image_b], label
    """

    # initialize pairs/label
    image_a = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    image_b = np.zeros((batch_size, np.shape(imgs)[1], np.shape(imgs)[2], np.shape(imgs)[3]))
    label = np.zeros(batch_size)
    
    # for each img
    for i in range(batch_size):

        # random index
        idx1 = random.randint(0, len(imgs) - 1)
        idx2 = random.randint(0, len(imgs) - 1)
        
        # postitive pairs
        if (i % 2 == 0):
            l = 1
            while (labels[idx1] != labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)    

        # negative pairs 
        else: 
            l = 0
            while (labels[idx1] == labels[idx2]):
                idx2 = random.randint(0, len(imgs) - 1)

        image_a[i,:,:,:] = imgs[idx1,:,:,:]
        image_b[i,:,:,:] = imgs[idx2,:,:,:]
        label[i] = l

    return [image_a, image_b], label 


def PairGenerator(imgs, labels, batch_size):
    '''
    This is a Generator. It will call GetSiameseData to create an infinite number of batches of pairs. 
    This is what we'll give to the network when it comes time to train the model. 
    '''
    while True:
        [image_a, image_b], label = GetSiameseData(imgs, labels, batch_size)

        # pass image through generator with labels
        genX1 = datagen.flow(image_a, label,  batch_size=batch_size, seed=42)
        genX2 = datagen.flow(image_a, image_b, batch_size=batch_size, seed=42)

        # get next batch, with augmentation
        X1i = genX1.next()
        X2i = genX2.next()

        #yield [image_a, image_b], label
        yield [X1i[0], X2i[1]], X1i[1]


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

left_input = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))
right_input = tf.keras.Input((IMG_SIZE,IMG_SIZE,3))

feature_1 = base_model(left_input)
feature_2 = base_model(right_input)

combined = layers.concatenate([feature_1, feature_2])
combined = layers.Dense(128, activation='relu')(combined)

output = layers.Dense(1,activation='sigmoid')(combined)
siamese_network = tf.keras.Model([left_input,right_input], output)

siamese_network.compile(loss=contrastive_loss, optimizer=keras.optimizers.Adam(learning_rate=0.00001), metrics=['accuracy'])

batch_size = 64
validation_size = len(x_val)  
num_epochs = 50
training_gen = PairGenerator(x_train, y_train, batch_size)
x_validation, y_validation = GetSiameseData(x_val, y_val, validation_size) 

siamese_network.fit(
    training_gen,
    steps_per_epoch = len(x_train) // batch_size, 
    batch_size=batch_size, 
    epochs=num_epochs, 
    shuffle=True, 
    validation_data=(x_validation, y_validation), 
)


# ========================= AUC ==========================================


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

probs, yprob = compute_probs(base_model, x_test, y_test)
fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
draw_roc(fpr, tpr, thresholds)

print(f"AUC: {auc}")



# ========================= CMC ==========================================


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

cmc = CMC(base_model, x_test)
print(cmc[:5])

# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(cmc)
ax.set_xlabel('Rank')
ax.set_ylabel('Count')
ax.set_title('CMC Curve')
plt.savefig(f"../results/{DATA}_contrastive_CMC.png", bbox_inches='tight')
