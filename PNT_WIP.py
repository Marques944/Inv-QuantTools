
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn import preprocessing
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.cluster import KMeans

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None):
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(train_df.columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
      return inputs, labels

  def make_dataset(self, data, train = False, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      
      # data = np.expand_dims(data,2)
      # print(data.shape)
      if train == True:
        np.expand_dims(data,2)

      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      
      # print(type(ds))
     
      ds = ds.map(self.split_window)

      # print(list(ds.as_numpy_iterator()))
      # exit()
      
      return ds


if __name__ == '__main__':
    # PREPARE DATA
    df = pd.read_excel('data_preprocessed.xlsx')
    df = df.set_index(df.Date)
    df0 = df.drop(columns = 'Date')
    df1 = df0.dropna()
    n = len(df1)

    # PREPARE DATA

    df1_cl = df1.iloc[:,0:2]

    dataset=[]


    for i in range(len(df1_cl)):
        dataset.append([df1_cl.iloc[i,0],df1_cl.iloc[i,1]])


    array = np.array(dataset)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(dataset)

    clusters = kmeans.cluster_centers_
    predict = []
    predict = kmeans.fit_predict(array)
    predict = predict

    df1.insert(2, 2, predict)
    df1.insert(3, 3, df1.iloc[:,0])

    # plt.scatter(array[df1[3] == 0,0],array[df1[3] == 0,1], s=50, color = 'red')
    # plt.scatter(array[y_km == 1,0],array[y_km == 1,1], s=50, color = 'blue')
    # plt.scatter(array[y_km == 2,0],array[y_km == 2,1], s=50, color = 'green')
    # plt.scatter(array[y_km == 3,0],array[y_km == 3,1], s=50, color = 'yellow')
    # plt.show()
    # exit()
    
    
    # hold out test data
    train_df1 = df1[0:int(0.8*n)]
    
    test_df1 = df1[int(0.8*n):]
    mm_scaler = preprocessing.StandardScaler()
    train_dfm = mm_scaler.fit_transform(train_df1)
    test_dfm = mm_scaler.transform(test_df1)
    train_df = pd.DataFrame(train_dfm, index=train_df1.index, columns=train_df1.columns)
    test_df = pd.DataFrame(test_dfm, index=test_df1.index, columns=test_df1.columns)
   
    # print(train_df.shape)
    # train_df = np.expand_dims(train_df,2)
    # print(train_df.shape)

    # train_df = pd.DataFrame(train_df)
    # define sliding window
    
    lf = 1      # look forward
    ks = 5      # kernel size
    lw = 1      # label width
    lb = 15

    batchsize = 75

    # useCNN = False
    # look back
    window = WindowGenerator(input_width=lb, label_width=lw, shift=1, input_columns=[0,1,2], label_columns=[3])

    td = window.make_dataset(train_df, train = True,  batchsize=batchsize, shuffle=True)
    # cross-validation

    train_data = td.take(10)
    val_data = td.skip(10)
    test_data = window.make_dataset(test_df)


    # SET-UP AND TRAIN MODEL
    model = tf.keras.Sequential()

    # Version 1: Convolutional Network

    # Inputshape parameter still causing issues
    
    model.add(Conv1D(batchsize, kernel_size=ks, activation='relu', padding="same", use_bias=False, input_shape = (3,1)))
    model.add(MaxPooling1D(pool_size=4, strides=2, padding='same'))
    model.add(Conv1D(filters=73, kernel_size=ks-1, activation='relu', use_bias=False))
    model.add(MaxPooling1D(pool_size=4, strides=2, padding='same'))
    # model.add(GlobalMaxPooling1D())
    # model.add(Flatten())
    model.add(tf.keras.layers.Dense(35,activation='relu', use_bias = True))
    model.add(tf.keras.layers.Dense(1))
    

    # # Version 2: Recurrent Network
    # model.add(tf.keras.layers.SimpleRNN(2, return_sequences=False, return_state=False, activation='tanh', use_bias=True))
    # model.add(tf.keras.layers.Dense(1, activation='tanh', use_bias=True))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=150, decay_rate=0.95, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError()])
    model.run_eagerly = False
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, mode='min')
    history = model.fit(train_data, validation_data=val_data, epochs=500, batch_size=75, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.legend(['training loss', 'validation loss'])
# 
    eval_train = window.make_dataset(train_df, batchsize=train_df.shape[0], shuffle=False)
    eval_test = window.make_dataset(test_df, batchsize=test_df.shape[0], shuffle=False)

    # CHECK IS and OS performance and P/L of a trading strategy
    plt.figure()
    plt.subplot(221)
    y_pred = model.predict(eval_train)
    y_true = np.concatenate([y for x, y in eval_train], axis=0)
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(train_df.index[lb+lf-1:], y_true[:, -1, -1])
    plt.plot(train_df.index[lb + lf - 1:], y_pred[:, -1], '--')
    plt.title('in-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred'])

    plt.subplot(222)
    y_mkt = train_df1.iloc[lb+lf-1:,:].loc[:,'u1']
    # position taking: simple switch
    pos = np.sign(np.squeeze(y_pred[:,  -1]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    sr = pnl.mean()/pnl.std() * np.sqrt(52)
    plt.title('in-sample Sharpe ratio = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])

    plt.subplot(223)
    y_pred = model.predict(eval_test)
    y_true = np.concatenate([y for x, y in eval_test], axis=0)
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(test_df.index[lb+lf-1:], y_true[:, -1, -1])
    plt.plot(test_df.index[lb+lf-1:], y_pred[:, -1], '--')
    plt.title('out-of-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred'])

    plt.subplot(224)
    y_mkt = test_df1.iloc[lb+lf-1:,:].loc[:,'u1']
    # position taking: simple switch
    pos = np.sign(np.squeeze(y_pred[:, -1]))
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    sr = pnl.mean()/pnl.std()  * np.sqrt(52)
    plt.title('out-of-sample Sharpe ratio = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])
    plt.show()
