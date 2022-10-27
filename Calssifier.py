
"""
Created on Thu Oct 27 10:59:37 2022

@author: ali.raza
"""

#------------------------------------------------Imports---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import stats
import tensorflow as tf
from sklearn import metrics
import itertools
#-----------------------------------------Important modules and Functions--------------------------------------------------------------------------



def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset-mu)/sigma


def windows(data,size):
    start = 0
    while start< data.count():
        yield int(start), int(start + size)
        start+= (size/2)
        
        
# segmenting the time series
def segment_signal(data, window_size = 140):
    segments = np.empty((0,window_size,9))
    labels= np.empty((0))
    for (start, end) in windows(data['ax'],window_size):
        ax = data['ax'][start:end]
        ay = data['ay'][start:end]
        az = data['az'][start:end]
        mx= data['mx'][start:end]
        my = data['my'][start:end]
        mz = data['mz'][start:end]
        gx= data['gx'][start:end]
        gy = data['gy'][start:end]
        gz= data['gz'][start:end]
        if(len(data['ax'][start:end])==window_size):
             segments = np.vstack([segments,np.dstack([ax,ay,az,mx,my,mz,gx,gy,gz])])
             labels = np.append(labels,stats.mode(data['label'][start:end])[0][0])
    return segments, labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#--------------------------------------------------------------Importing and Preprocessing data----------------------------------------------------
raw_data= pd.read_csv('dataactivity.csv', sep=';')
x_train=raw_data            

segments, labels = segment_signal(x_train)  

segments_normalised=featureNormalize(segments)
x_train, x_test, target_train, target_test=train_test_split(segments_normalised, labels, test_size=0.1,shuffle=True)


y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

#----------------------------------------------------Hyper-parameters---------------------------------------------------------------------------
num_classes = 15
input_shape =(90,9)
learning_rate = 0.01
weight_decay = 0.001
batch_size = 56
num_patches = 1
projection_dim = 6
num_heads =5
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
mlp_head_units = [20]  # Size of the dense layers of the final classifier



#----------------------------------------------------- Classifier Modeling-----------------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
    ],
    name="data_augmentation",
)

# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=6,name=None,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
    def build(self,input_shape):
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim )
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches,
                })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    
    
    
        
def create_vit_classifier():
    inputs = layers.Input(shape=(x_train.shape[1],9))
    # Augment data.
    augmented = data_augmentation(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(augmented)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.2
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.2)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    #features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes,activation='softmax')(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


#--------------------------------------------Creat and Compile classififer----------------------------------------------------------------
classifier = create_vit_classifier()
optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
classifier.compile(
        optimizer='adam',
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy",
        ],
    )
classifier.summary()


optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

#-----------------------------------------Model Fittinh--------------------------------------------------------------------------





history = classifier.fit(
        x=x_train,
        y=y_train,
        batch_size=30,
        epochs=50,
        shuffle=True,
        validation_data=(x_test,y_test) )



#--------------------------------------- Testing Performance-----------------------------------------------------------------------
x_test_pred = classifier.predict(x_test)
        #decoding
y_pred=np.argmax(x_test_pred, axis=1)
y_orig=np.argmax(y_test, axis=1)

           
        

target_names = list([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
clf_report = classification_report(y_orig, y_pred, target_names=target_names,output_dict=True)
plt.show(sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True))   
print(classification_report(y_orig, y_pred))
score = metrics.accuracy_score(y_orig, y_pred)
print("accuracy:  %0.3f" % score)

cm = metrics.confusion_matrix(y_orig, y_pred)
plt.figure(figsize=(15, 15))
plot_confusion_matrix(cm, classes=['Standing',
'Sitting',
'Walking',
'joging',
'upstair',
'downstair',
'Eating',
'Writing',
'Using laptop',
'washing face',
'Washing hand',
'swiping',
'vacuming',
'dusting a surface',
'Brushing Teeth'])
plt.show()
      