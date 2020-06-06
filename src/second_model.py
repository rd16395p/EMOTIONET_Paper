import numpy as np
#from keras.models import load_model
import tensorflow as tf
import pandas as pd

df_eval_y = 'valdiationlabels.csv'
df_eval_x = 'validationdata.npy'

#X_test = pd.read_csv(df_eval_x, header=None).values#.transpose()
y_test = pd.read_csv(df_eval_y, header=None).values.ravel()#.transpose()
X_test = np.load(df_eval_x)
#X_test = np.asarray(X_test, dtype=np.float)
y_test = np.asarray(y_test, dtype=np.int32)
#X_test = np.asarray(X_test, dtype=np.float)
print(X_test.dtype, y_test.dtype)
print(X_test.shape, y_test.shape)

model = tf.keras.models.load_model('cnn1_model.h5')

result = model.predict_classes(X_test, verbose=1)

print(result)
print(type(result))
#score = model.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
#print(result[0])

indexnumber = list(result)
indexnumber = [i for i, x in enumerate(indexnumber) if x == 1]
print(indexnumber)

X_test_new = np.copy(X_test[indexnumber, ])
y_test_new = np.copy(y_test[indexnumber, ])

model2 = tf.keras.models.load_model('cnn2_model.h5')

result_2 = model2.predict_classes(X_test, verbose=1)
print(result_2)
#score = model2.evaluate(X_test, y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
