import numpy as np
import pandas as pd

df_train_x = 'trainimages.csv'
df_label = 'largetrainlabels.csv'
df_test_x = 'testimages.csv'
df_label_eval = 'largetestlabels.csv'

answer = input("Would you like to save data or labels?")
answer2 = input("How many splits?")

if answer == "data":
    print("Loading data....")
    X_train = pd.read_csv(df_train_x, header=None)
    X_test = pd.read_csv(df_test_x, header=None)
    y_train = pd.read_csv(df_label, header=None).values #.ravel()
    #y_train = np.asarray(y_train, dtype=np.int32)
    y_test = pd.read_csv(df_label_eval, header=None).values.ravel()
    #y_test = np.asarray(y_test, dtype=np.int32)
    X_train = np.asarray(X_train, dtype=np.float32)
    #X_test = np.asarray(X_test, dtype=np.float)
    #X_test = np.asarray(X_test,dtype=np.float32)
    print(X_train.shape)#, X_train.dtype)
    #print(y_train.shape)#, y_train.dtype)
    #print(X_test.shape, X_test.dtype)
    #print(y_test.shape, y_test.dtype)
    #print(type(X_test))
    #print(type(y_test))
    print("Data loaded!")

    train_splits = np.array_split(X_train, int(answer2))

    i = 1
    for x in train_splits:
        #df = pd.DataFrame(x)
        save('X_train_'+str(i)+'.npy',x)
        i += 1
else:
    print("Loading data....")
    y_train = pd.read_csv(df_label, header=None).values #.ravel()
    #y_train = np.asarray(y_train, dtype=np.int32)
    y_test = pd.read_csv(df_label_eval, header=None).values.ravel()
    #y_test = np.asarray(y_test, dtype=np.int32)
    #X_test = np.asarray(X_test, dtype=np.float)
    #X_test = np.asarray(X_test,dtype=np.float32)
    print(y_train.shape)#, y_train.dtype)
    #print(y_test.shape, y_test.dtype)
    #print(type(X_test))
    #print(type(y_test))
    print("Data loaded!")

    train_splits = np.array_split(y_train, int(answer2))

    i = 1
    for x in train_splits:
        #df = pd.DataFrame(x)
        save('y_train_'+str(i)+'.npy',x)
        i += 1
