import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing tools
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Models
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv1D, AveragePooling1D, Flatten, Dropout


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


def verify_target(target):
    if 0 in np.unique(target):
        # Water is present
        return 0

    else:
        # Water is not present
        return 1


def window_data(df, target, window_size=20, overlap=0):
    windowed_data = []
    new_target = []
    start_index = 0
    while True:
        if start_index + window_size > len(df)-1:
            break

        windowed_data.append(np.array(df[start_index : (start_index + window_size)]))
        new_target.append(verify_target(target[start_index : (start_index + window_size)]))

        start_index = start_index + (window_size - overlap)
        # print(len(df[start_index : (start_index + window_size)]))

    return np.stack(windowed_data), np.array(new_target)


def get_undersampled_data(data, target):
    water_idx = np.where(target == 0)[0]

    not_water_idx = np.where(target == 1)[0]
    np.random.shuffle(not_water_idx)
    not_water_idx = not_water_idx[:len(water_idx)]

    water = data[water_idx]
    water_y = target[water_idx]

    not_water = data[not_water_idx]
    not_water_y = target[not_water_idx]

    # print(water.shape)
    # print(not_water.shape)

    new_data = np.concatenate((water, not_water), axis=0)
    new_target = np.concatenate((water_y, not_water_y), axis=0)

    tmp = np.arange(len(new_data))
    np.random.shuffle(tmp)

    new_data = new_data[tmp]
    new_target = new_target[tmp]

    return new_data, new_target


def split_windowed_data(windowed_x, windowed_y, split=.7, overlap=0, shuffle=False):

    stop_index = int(round(len(windowed_x) * split)) - int(round(overlap/2))
    start_index = int(round(len(windowed_x) * split)) + int(round(overlap/2))

    train_x = windowed_x[:stop_index]
    train_y = windowed_y[:stop_index]

    test_x = windowed_x[start_index:]
    test_y = windowed_y[start_index:]

    return train_x, train_y, test_x, test_y


def make_model(input):
    # Input layer has N x k neurons, where k denotes the variate number of input time series and N denotes the length of
    # each univariate series

    # Convolutional layer has m filters with stride s and filter size k x l, where k denotes the variate number of the
    # time series in the preceding layer and l denotes the length of filter. A nonlinear transformation function f also
    # needs to be determined in this layer

    # Pooling layer has a feature map divided ino N equal length segments and every segment is represented by the
    # average or maximum value

    # Feature layer represents the time series as a series of feature maps.

    # Output layer has n neurons corresponding to n classes.

    # 2 Convolutional layers and two pooling layers
    # Input and output layers are data dependent
    # Use sigmoid for the activation function
    # Uses MSE
    # Filter size first conv layer 7, Filter size second conv layer 7
    # Mean-pooling size 3,
    # Filter number for first conv layer 6, filter number for second conv layer 12

    model = keras.Sequential()
    # model.add(Input(shape=(input.shape[1:])))
    model.add(Conv1D(filters=6, kernel_size=7, padding="same", activation="relu", input_shape=(input.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(AveragePooling1D(pool_size=3))
    model.add(Conv1D(filters=12, kernel_size=7, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(AveragePooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=['accuracy'])
    # model.summary()

    return model


def get_model(x_train, y_train): #, y_train, x_val, y_val, x_test, y_test):
    model = make_model(x_train)

    history = model.fit(x_train, y_train, epochs=100, verbose=0, validation_split=0.1, callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ])

    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.show()

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
    X = df['Reflectivity']            # Data for classification
    y = df['Crop']

    print(y)

    # Encode target value
    y = encode_data(y)

    # Standardize the data
    standardize = False
    if standardize == True:
        X_mean = X.mean()
        X_std = X.std()

        X = (X - X_mean) / X_std

    print(X.head())

    print("----------------------------------------------")

    scores = []
    cm = []
    cr = []
    # (100, 0), (50, 0), (25, 0), #(5, 0),
    # (100, 25), (50, 14), (25, 7), #(5, 2),
    # (100, 50), (50, 25), (25, 14),
    # (100, 75), (50, 38), (25, 19), #(5, 4),
    window_values = [(300, 299), (200, 199), (100, 99), (50, 49), (25, 24)]

    window_values = [(30, 29)]

    #window_values = [(250, 249), (250, 245), (230, 229), (230, 225)]

    for window_size, overlap in window_values:
        print(f"({window_size}, {overlap})")

        # Window data
        # window_size = 36
        # overlap = 35
        x_windowed, y_windowed = window_data(X.values, y, window_size, overlap)
        # print(y_windowed)

        undersample = True
        if undersample:
            x_windowed, y_windowed = get_undersampled_data(x_windowed, y_windowed)

        # Split data
        len_data = len(X)
        split_size = int(round(len_data * .7))
        # print(split_size)

        x_train, y_train, x_test, y_test = split_windowed_data(x_windowed, y_windowed, split=.8, overlap=overlap) #, shuffle=True)

        model = get_model(x_train, y_train)

        loss, accuracy = model.evaluate(x_test, y_test)

        print(type(x_test))
        print('------------------------------')
        print(x_test.shape)

        y_pred_prob = model.predict(x_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Compute the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

        # Compute the AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)

        # Print AUC score
        print(f"AUC: {roc_auc:.2f}")

        # Plot the ROC curve
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc='lower right')
        # plt.show()

        print("Test Accuracy: " + str(accuracy))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print()
        print()

        scores.append(accuracy)
        cm.append(confusion_matrix(y_test, y_pred))
        cr.append(classification_report(y_test, y_pred))

    return model

    # for i in range(len(window_values)):
    #     print()
    #     print(window_values[i])
    #     print(scores[i])
    #     print(cm[i])
    #     print(cr[i])
    #     print()
    #     print("---------------------------------------------")





# -----------------------------------------------------

    # Scale data
    # scale = True
    # if scale == True:
    #     scaler = StandardScaler()
    #     tmp = reshape_input_data(df['Reflectivity'])
    #     X_scaled = scaler.fit_transform(tmp)
    #     X_class = pd.DataFrame(X_scaled)
    #
    # # Split data
    # X_train, X_test, Y_train, Y_test = train_test_split(X_class, y, test_size=.1, shuffle=True, stratify=y)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=.5, shuffle=True, stratify=Y_test)
    #
    # # Reshape for models
    # X_train = reshape_input_data(X_train)
    # X_val = reshape_input_data(X_val)
    # X_test = reshape_input_data(X_test)


if __name__=="__main__":
    main()

    """
    Need to make it so that the training set can randomly shuffle.
    Can try SMOTE-TS
    Can try Undersampling
    Can a dropout layers
    L1/L2 regularization
    ROC-AUC: Measures the model's ability to distinguish classes
    Maybe a course grid search across model parameters
    """

