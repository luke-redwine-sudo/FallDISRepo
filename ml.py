import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Preprocessing tools
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Clustering
from sklearn.cluster import KMeans


def read_data(dir_path):
    seasons = ['Spring', 'Fall']

    read_df = pd.DataFrame()

    # Create a figure
    plt.figure(figsize=(10, 6))

    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'black', 'cyan', 'magenta']
    i = 0

    # Loop over all seasons
    for season in seasons:
        #if season == 'Fall':
        #    continue

        print(season)
        sub_dir_path = dir_path + season + ' Data/'

        # Loop over all dates
        for date_dir in os.listdir(sub_dir_path):
            print(sub_dir_path)
            print(date_dir)

            # Read data
            try:
                new_df = pd.read_csv(sub_dir_path + date_dir + '/' + 'processed_data.csv')
                print('\t' + str(new_df['DateTime'][0]))

                new_df = new_df[new_df["Alt"] > 13]
                df_water = new_df[new_df['Crop'] == 'water']
                df_not_water = new_df[new_df['Crop'] != 'water']

                sns.kdeplot(df_water['Reflectivity'].to_numpy(), color=colors[i])
                sns.kdeplot(df_not_water['Reflectivity'].to_numpy(), color=colors[i], linestyle='--')

                read_df = pd.concat([read_df, new_df])
                i+=1

            except:
                print('processed_data.csv does not exist for ' + date_dir)
                pass



    # Plot KDE only
    # sns.kdeplot(df['Reflectivity'].to_numpy(), color='green')
    # sns.kdeplot(df_water['Reflectivity'].to_numpy(), color='blue')
    # sns.kdeplot(df_not_water['Reflectivity'].to_numpy(), color='red')

    # Add labels and title
    plt.title('KDE Plot Only')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Show the plot
    #plt.show()

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


def get_decision_tree(x_train, y_train, x_test, y_true, class_weight):
    print('\n---------------- Decision Tree Results ----------------\n')

    # Create and train decision tree
    clf = tree.DecisionTreeClassifier(class_weight=class_weight)
    clf = clf.fit(x_train, y_train)

    # print(tree.export_text(clf))

    # Predict using trained decision tree
    y_pred = clf.predict(x_test)

    print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred))

    return clf


def get_decision_tree_cv(x, y, splits=5, shuffle=True, max_depth=None, class_weight=None):
    print('\n---------------- Decision Tree CV Results ----------------\n')

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = tree.DecisionTreeClassifier(class_weight=class_weight)

    # Reshape the data for training
    # x = reshape_input_data(x)

    # Train model using cross-validation and n folds
    scores = cross_val_score(model, x, y, cv=kfolds, scoring='accuracy')

    print(scores)


def get_extra_tree(x_train, y_train, x_test, y_true, spliter='random'):
    print('\n---------------- Extra Tree Results ----------------\n')

    # Create and train decision tree
    clf = tree.ExtraTreeClassifier(splitter=spliter)
    clf = clf.fit(x_train, y_train)

    # print(tree.export_text(clf))

    # Predict using trained decision tree
    y_pred = clf.predict(x_test)

    print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred))

    return clf


def get_random_forest(x_train, y_train, x_test, y_true, class_weight=None):
    print('\n---------------- Random Forest Results ----------------\n')

    # Create and train decision tree
    clf = RandomForestClassifier(class_weight=class_weight)
    clf.fit(x_train, y_train)

    # Predict using trained decision tree
    y_pred = clf.predict(x_test)

    print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred))

    return clf


def get_random_forest_cv(x, y, splits=5, shuffle=True, num_trees=100, max_depth=None, class_weight=None):
    print('\n---------------- Random Forest CV Results ----------------\n')

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = RandomForestClassifier(class_weight=class_weight)

    # Reshape the data for training
    # x = reshape_input_data(x)

    # Train model using cross-validation and n folds
    scores = cross_val_score(model, x, y, cv=kfolds, scoring='accuracy')

    print(scores)


def get_random_forest_gs(x_train, y_train, x_test, y_true, splits=3, shuffle=True, num_trees=[10, 50, 100, 500],
                         max_depth=[10, 20], class_weight=None):
    print('\n---------------- Random Forest Grid Search Results ----------------\n')

    # Number of trees 10, 50, 100, 500)
    # Max depth 3, 4, 10, 20

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = RandomForestClassifier(class_weight=class_weight)

    # Reshape the data for training
    # x_train = reshape_input_data(x_train)

    clf = GridSearchCV(estimator=model, param_grid={'n_estimators': num_trees, 'max_depth': max_depth}, cv=kfolds,
                       scoring='accuracy', verbose=1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred))


def get_kmeans(x_train, y_train, x_test, y_true, xt_test, water, not_water):
    print('\n---------------- KMeans Results ----------------\n')

    # Create and train decision tree
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x_train)

    # Predict using trained decision tree
    y_pred = kmeans.predict(x_test)

    print(y_pred)

    color_df = pd.DataFrame(y_pred)
    color_df.replace(0, 'red', inplace=True)
    color_df.replace(1, 'blue', inplace=True)

    color_df.columns = ['Color']

    # color_df['LAT'] = xt_test['Adjust_LAT_M']
    print(color_df.head())
    xt_test = xt_test.set_index(np.arange(len(xt_test)))
    # print(xt_test['Adjust_LAT_M'])

    color_df['Reflectivity'] = xt_test['Reflectivity']
    color_df['Reflectivity'] = xt_test['Reflectivity']

    # color_df['LAT'] = xt_test['Lat']
    # color_df['LON'] = xt_test['Lng']

    print(color_df.head())

    all_blue = color_df[color_df['Color'] == 'blue']
    all_red = color_df[color_df['Color'] == 'red']

    # Create a figure and a 1x2 grid of subplots (2 subplots in one row)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # First subplot: scatter plot of x1 vs y1
    axs[0, 0].scatter(all_blue['Reflectivity'], all_blue['Reflectivity'], color='blue', label='Dataset 1')
    axs[0, 0].scatter(all_red['Reflectivity'], all_red['Reflectivity'], color='red', label='Dataset 1')
    axs[0, 0].set_title('Scatter Plot 1')
    axs[0, 0].set_xlabel('X1')
    axs[0, 0].set_ylabel('Y1')
    axs[0, 0].legend()

    # Second subplot: scatter plot of x1 vs y1
    axs[1, 0].scatter(all_red['Reflectivity'], all_red['Reflectivity'], color='red', label='Dataset 1')
    axs[1, 0].scatter(all_blue['Reflectivity'], all_blue['Reflectivity'], color='blue', label='Dataset 1')
    axs[1, 0].set_title('Scatter Plot 1')
    axs[1, 0].set_xlabel('X1')
    axs[1, 0].set_ylabel('Y1')
    axs[1, 0].legend()

    # Third subplot: scatter plot of x1 vs y1
    axs[0, 1].scatter(water['Reflectivity'], water['Reflectivity'], color='blue', label='Dataset 1')
    axs[0, 1].scatter(not_water['Reflectivity'], not_water['Reflectivity'], color='red', label='Dataset 1')
    axs[0, 1].set_title('Scatter Plot 1')
    axs[0, 1].set_xlabel('X1')
    axs[0, 1].set_ylabel('Y1')
    axs[0, 1].legend()

    # Fourth subplot: scatter plot of x1 vs y1
    axs[1, 1].scatter(not_water['Reflectivity'], not_water['Reflectivity'], color='red', label='Dataset 1')
    axs[1, 1].scatter(water['Reflectivity'], water['Reflectivity'], color='blue', label='Dataset 1')
    axs[1, 1].set_title('Scatter Plot 1')
    axs[1, 1].set_xlabel('X1')
    axs[1, 1].set_ylabel('Y1')
    axs[1, 1].legend()

    # plt.scatter(all_blue['LAT'], all_blue['LON'], color='blue', alpha=0.2)
    # plt.scatter(all_red['LAT'], all_red['LON'], color='red', alpha=0.2)

    # plt.scatter(x_test, x_test, cmap={'blue', 1: 'red'})
    #plt.show()

# Visualizations
# 1. Scores
# 2. Confusion Matrix
# 3. Decision Tree visualization
# 4. Precision Recall Curve
# 5. ROC Curve
# 6. Clustering visualization
# 7. "Map" with clustered colors



def main():
    # Path to highest level data directory
    # dir_path = 'D:/redwi/Documents/Thesis Data/'
    dir_path = 'D:/redwi/Documents/Thesis Data/Data - Copy/'

    # Read in data from directories
    df = read_data(dir_path)

    df_water = df[df['Crop'] == 'water']
    df_not_water = df[df['Crop'] != 'water']



    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot KDE only
    # sns.kdeplot(df['Reflectivity'].to_numpy(), color='green')
    sns.kdeplot(df_water['Reflectivity'].to_numpy(), color='blue')
    sns.kdeplot(df_not_water['Reflectivity'].to_numpy(), color='red')

    # Add labels and title
    plt.title('KDE Plot Only')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Show the plot
    #plt.show()

    # Fill empty values
    # df = clean_data(df)

    # Undersample data
    undersample = True
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

    # print(df[df['Crop'] == ''])
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

    for i in np.arange(.1, .2, .1):
        print('Test size' + str(i))

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X_class, y, test_size=i, shuffle=True, stratify=y)

        # Compute class weight
        cw = False
        if cw == True:
            class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
            class_weight_dict = dict(zip(np.unique(Y_train), class_weight))
        else:
            class_weight_dict = None

        # Reshape for models
        X_train = reshape_input_data(X_train)
        X_test = reshape_input_data(X_test)

        # Balance dataset using SMOTE
        smote = True
        if smote == True:
            smote = SMOTE()
            X_train, Y_train = smote.fit_resample(X_train, Y_train)

        # Call models
        _ = get_decision_tree(X_train, Y_train, X_test, Y_test, class_weight=class_weight_dict)
        d = _

        _ = get_extra_tree(X_train, Y_train, X_test, Y_test, 'random')

        _ = get_random_forest(X_train, Y_train, X_test, Y_test, class_weight=class_weight_dict)
        r = _

    # exit()

# --------------------------------------------------------------------------------------------
    # Cross-Validation

        # Balance dataset using SMOTE
    smote = True
    if smote == True:
        smote = SMOTE()
        X_train, Y_train = smote.fit_resample(X_train, Y_train)

    get_decision_tree_cv(X_train, Y_train, splits=5, class_weight=class_weight_dict)

    get_random_forest_cv(X_train, Y_train, splits=5, class_weight=class_weight_dict)

# --------------------------------------------------------------------------------------------
    # Course Grid Search

    get_random_forest_gs(X_train, Y_train, X_test, Y_test, class_weight=class_weight_dict)

    # exit()

# --------------------------------------------------------------------------------------------
    # Clustering

    X_train_clust, X_test_clust, Y_train_clust, Y_test_clust = train_test_split(X_clust, y, test_size=.2, shuffle=True)

    x_train_clust = reshape_input_data(X_train_clust['Reflectivity'])
    x_test_clust = reshape_input_data(X_test_clust['Reflectivity'])

    get_kmeans(x_train_clust, Y_train, x_test_clust, Y_test, X_test_clust, df_water, df_not_water)

# -------------------------------------------------------------------------------------------
    # 'Deep' Learning Model

    return r, d



if __name__=="__main__":
    main()