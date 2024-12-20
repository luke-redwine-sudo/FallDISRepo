import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing tools
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report

# Models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


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
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    return model


def get_NN(x_train, y_train, x_val, y_val, x_test, y_test):
    model = make_model(x_train)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(x_test, y_test)

    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    print('Accuracy: ' + str(accuracy))
    print('Loss: ' + str(loss))

    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['accuracy'], label='Val Loss')
    plt.legend()
    #plt.show()

    return model


def main():
    # Path to highest level data directory
    # dir_path = 'D:/redwi/Documents/Thesis Data/'
    dir_path = 'D:/redwi/Documents/Thesis Data/Data - Copy/'

    # Read in data from directories
    df = read_data(dir_path)

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
    X_class = df['Reflectivity']            # Data for classification
    X_clust = df.drop(['Crop'], axis=1)     # Data for clustering
    y = df['Crop']

    # Encode target value
    y = encode_data(y)

    # Scale data
    scale = True
    if scale == True:
        scaler = StandardScaler()
        tmp = reshape_input_data(df['Reflectivity'])
        X_scaled = scaler.fit_transform(tmp)
        X_class = pd.DataFrame(X_scaled)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_class, y, test_size=.1, shuffle=True, stratify=y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.5, shuffle=True, stratify=Y_test)

    # Reshape for models
    X_train = reshape_input_data(X_train)
    X_val = reshape_input_data(X_val)
    X_test = reshape_input_data(X_test)

    # Balance dataset using SMOTE
    smote = True
    if smote == True:
        smote = SMOTE()
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

# -------------------------------------------------------------------------------------------
# 'Deep' Learning Model
    nn = get_NN(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    return nn

if __name__=="__main__":
    main()