from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import tensorflow_datasets as tfds
import numpy as np

tpu_address = input("What is the name of the TPU address?")

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + tpu_address)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Reshape((256, 256, 1)),
       tf.keras.layers.Conv2D(32, 2, activation='relu', input_shape=(-1, 256, 256, 1)),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=4),
       tf.keras.layers.Conv2D(92, 2, activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
       tf.keras.layers.Conv2D(182, 2, activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(1024, activation='relu'),
       tf.keras.layers.Dropout(rate=.4, noise_shape=None, seed=None),
       tf.keras.layers.Dense(2)])

def get_dataset(batch_size=200):
  print("loading train data....")
 # features = np.load("X_train_1.npy")
  #labels = np.zeros(shape=(1000,1))
  #labels = tf.cast(labels, tf.float32)
  #labels = np.load("test_labels.npy")
  #train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  
  features_name = "X_train_"
  features = np.load("/home/rydagostino/data/train2/X_train_1.npy")
  filenames = []
  for i in range(2,7,1):
    tmp = np.load("/home/rydagostino/data/train2/"+features_name+str(i)+".npy")
    features = np.concatenate((features,tmp))  	
       #filenames.append(os.getcwd()+"/"+features_name+str(i)+".npy")
  
  labels_name = "y_train_"
  labels = np.load("/home/rydagostino/data/train2/y_train_1.npy")
  filenames = []
  for i in range(2,7,1):
    tmp = np.load("/home/rydagostino/data/train2/"+labels_name+str(i)+".npy")
    labels = np.concatenate((labels,tmp)) 
 # labels = np.zeros(shape=(7000,1))
  #labels[4000:7000] = 1
  #features = np.zeros(shape=(7000,65536))

  #labels = labels.astype('float32')
  train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  
  train_dataset = train_dataset.shuffle(1000).batch(batch_size)

  features_name = "X_test_"
  features = np.load("/home/rydagostino/data/test2/X_test_1.npy")
  #filenames = []
  #for i in range(2,7,1):
   # tmp = np.load("/home/rydagostino/data/test2/"+features_name+str(i)+".npy")
    #features = np.concatenate((features,tmp))       
       #filenames.append(os.getcwd()+"/"+features_name+str(i)+".npy")

  labels_name = "y_test_"
  labels = np.load("/home/rydagostino/data/test2/y_test_1.npy")
 #filenames = []
  #for i in range(2,7,1):
   # tmp = np.load("/home/rydagostino/data/test2/"+labels_name+str(i)+".npy")
   # labels = np.concatenate((labels,tmp))

#test_dataset = tf
  test_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
  
  test_dataset = test_dataset.shuffle(1000).batch(batch_size)

  return train_dataset, test_dataset

strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = create_model()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

train_dataset, test_dataset = get_dataset()

checkpoint_path = "gs://tpu_models_1/model_2/"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(train_dataset,
          epochs=40,
          validation_data=test_dataset,
          callbacks=[cp_callback])

model.save('cnn2_model.h5') 
