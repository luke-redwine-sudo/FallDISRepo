# Anomaly Detection

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing tools
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report

# Models
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, Conv1DTranspose


def read_data(dir_path):
    seasons = ['Spring', 'Fall']

    read_df = pd.DataFrame()

    # Loop over all seasons
    for season in seasons:
        print(season)
        sub_dir_path = dir_path + season + ' Data/'

        # Loop over all dates
        for date_dir in os.listdir(sub_dir_path):
            print(date_dir)

            # Read data
            try:
                new_df = pd.read_csv(sub_dir_path + date_dir + '/' + 'processed_data.csv')
                print('\t' + str(new_df['DateTime'][0]))
                read_df = pd.concat([read_df, new_df])

            except:
                print('processed_data.csv does not exist for ' + date_dir)
                pass

    return read_df


def clean_data(input_df):
    # Get only the necessary columns
    input_df = input_df[['Reflectivity', 'Crop']]

    # Fill in empty values
    input_df = input_df.fillna('not_water')

    return input_df


def encode_data(target_values):
    # Can switch to One Hot Encoding later
    le = LabelEncoder()
    encoded_data = le.fit_transform(target_values)

    return encoded_data


def reshape_input_data(input_x):
    input_x = input_x.to_numpy()

    return input_x.reshape(-1, 1)


def make_model(input):
    kernel_size = 6

    model = keras.Sequential()
    model.add(Input(shape=(input.shape[1], input.shape[2])))
    model.add(Conv1D(filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv1D(filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"))
    model.add(Conv1DTranspose(filters=16, kernel_size=kernel_size, padding="same", strides=2, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Conv1DTranspose(filters=32, kernel_size=kernel_size, padding="same", strides=2, activation="relu"))
    model.add(Conv1DTranspose(filters=1, kernel_size=kernel_size, padding="same"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    # model = Sequential([
    #     Dense(32, activation='relu', input_shape=(input.shape[1],)),
    #     Dense(16, activation='relu'),
    #     Dense(8, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    return model


def get_ae(x_train, test): #, y_train, x_val, y_val, x_test, y_test):
    model = make_model(x_train)

    print(x_train.shape)

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history = model.fit(x_train, x_train, epochs=20, batch_size=128, verbose=1, validation_split=0.1, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ])

    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.show()

    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    threshold = np.max(train_mae_loss)
    print(threshold)

    # plt.hist(train_mae_loss, bins=50)
    # plt.show()
    #
    # plt.plot(x_train[0])
    # plt.plot(x_train_pred[0])
    # plt.show()

    return model, threshold

    exit()

    loss, accuracy = model.evaluate(x_test, y_test)

    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    print('Accuracy: ' + str(accuracy))
    print('Loss: ' + str(loss))

    # plt.figure()
    # plt.plot(history.history['loss'], label='Loss')
    # plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.legend()
    # #plt.show()
    #
    # plt.figure()
    # plt.plot(history.history['loss'], label='Loss')
    # plt.plot(history.history['accuracy'], label='Val Loss')
    # plt.legend()
    # #plt.show()

    return model


def main():
    # Path to highest level data directory
    # dir_path = 'D:/redwi/Documents/Thesis Data/'
    dir_path = 'D:/redwi/Documents/Thesis Data/Data - Copy/'

    # Read in data from directories
    df = read_data(dir_path)
    print(df.columns)

    # Fill empty values
    # df = clean_data(df)

    # Undersample data
    undersample = False
    if undersample == True:
        num_samples = len(df[df['Crop'] == 'water'])

        df_water = df[df['Crop'] == 'water']
        df_not_water = df[df['Crop'] != 'water']
        df_sampled = df_not_water.sample(n=num_samples)
        print(num_samples)
        print(len(df_sampled))

        df = pd.concat([df_water, df_sampled])
        print(len(df))
        print(len(df[df['Crop'] == 'water']))
        print(len(df[df['Crop'] != 'water']))

    # Split into data and target values
    X_class = df[["DateTime", 'Reflectivity', 'Crop']]            # Data for classification
    y = df['Crop']

    X_class = X_class.set_index('DateTime')

    X_class.head()
    # exit()

    # Encode target value
    y = encode_data(y)

    # Scale data
    scale = False
    if scale == True:
        scaler = StandardScaler()
        tmp = reshape_input_data(df['Reflectivity'])
        X_scaled = scaler.fit_transform(tmp)
        X_class = pd.DataFrame(X_scaled)

    # Split data
    train, og_test = train_test_split(X_class, test_size=.3, shuffle=False)

    print(og_test["Crop"])
    print(og_test[og_test["Crop"] == 'water'])

    train = train.drop(['Crop'], axis=1)
    test = og_test.drop(['Crop'], axis=1)
    # X_train, X_test, Y_train, Y_test = train_test_split(X_class, y, test_size=.1, shuffle=False)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.5, shuffle=False)

    # Standardize the data
    standardize = True
    if standardize == True:
        train_mean = train.mean()
        train_std = train.std()

        train = (train - train_mean) / train_std

    window_size = 20
    def window_data(df, window_size=20, overlap=0):
        windowed_data = []
        start_index = 0
        while True:
            if start_index + window_size > len(df)-1:
                break

            windowed_data.append(df[start_index : (start_index + window_size)])
            start_index = start_index + (window_size - overlap)
            # print(len(df[start_index : (start_index + window_size)]))

        return np.stack(windowed_data)

    x_train = window_data(train.values, 36, 35)

    #
    # print(train.head())
    # print()
    # print(test.head())
    # exit()

    # Reshape for models
    # x_train = reshape_input_data(x_train)
    # X_val = reshape_input_data(X_val)
    # X_test = reshape_input_data(X_test)

    # Balance dataset using SMOTE
    # smote = False
    # if smote == True:
    #     smote = SMOTE()
    #     X_train, Y_train = smote.fit_resample(X_train, Y_train)

    # -------------------------------------------------------------------------------------------
    # 'Deep' Learning Model
    ae, threshold = get_ae(x_train, test) #X_train, Y_train, X_val, Y_val, X_test, Y_test)

    df_test_value = (test - train_mean) / train_std

    # print(test)

    print("Window test data")
    x_test = window_data(df_test_value.values, 36, 35)
    x_og_test = window_data(og_test.values, 36, 35)

    x_test_pred = ae.predict(x_test)

    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    # plt.hist(test_mae_loss)
    # plt.show()

    # threshold = .4
    anomalies = test_mae_loss > threshold

    print("Number of anomaly samples: ", np.sum(anomalies))
    # print("Indices of anomaly samples: ", np.where(anomalies))

    # for i in np.where(anomalies):
    #     # print(x_test[i])
    #     print(i)
    #     print(x_og_test[i].shape)
    #
    #     new_df = pd.DataFrame(x_og_test[i][0])
    #     print(new_df.head())
    # #     break

    not_anomalies = test_mae_loss <= threshold

    missed = 0
    ind = np.where(not_anomalies)
    for n in x_og_test[ind[0]]:
        for k in n:
            if k[1] == 'water':
                # print(k)
                missed += 1
        # print(n)

    print(missed)
    # print(x_og_test[ind[0][0][1]])
    #
    # missed = 0
    # for i in np.where(not_anomalies):
    #     new_df = pd.DataFrame(x_og_test[i])
    #     print(new_df)
    #     # print(new_df[new_df[1] == 'water'])
    #     # if (new_df[1] == 'water').sum() > 0:
    #     #     missed += 1

    # print(missed)


    return ae

if __name__=="__main__":
    main()
