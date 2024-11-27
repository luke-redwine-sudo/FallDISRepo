import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Preprocessing tools
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder, OneHotEncoder

# Models
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Clustering
from sklearn.cluster import KMeans


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


def get_decision_tree(x_train, y_train, x_test, y_true):
    print('\n---------------- Decision Tree Results ----------------\n')

    # Create and train decision tree
    clf = tree.DecisionTreeClassifier()
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


def get_decision_tree_cv(x, y, splits=5, shuffle=True, max_depth=None):
    print('\n---------------- Decision Tree CV Results ----------------\n')

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = tree.DecisionTreeClassifier()

    # Reshape the data for training
    x = reshape_input_data(x)

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


def get_random_forest(x_train, y_train, x_test, y_true):
    print('\n---------------- Random Forest Results ----------------\n')

    # Create and train decision tree
    clf = RandomForestClassifier()
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


def get_random_forest_cv(x, y, splits=5, shuffle=True, num_trees=100, max_depth=None):
    print('\n---------------- Random Forest CV Results ----------------\n')

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = RandomForestClassifier()

    # Reshape the data for training
    x = reshape_input_data(x)

    # Train model using cross-validation and n folds
    scores = cross_val_score(model, x, y, cv=kfolds, scoring='accuracy')

    print(scores)


def get_random_forest_gs(x_train, y_train, x_test, y_true, splits=3, shuffle=True, num_trees=[10, 50, 100, 500],
                         max_depth=[10, 20]):
    print('\n---------------- Random Forest Grid Search Results ----------------\n')

    # Number of trees 10, 50, 100, 500)
    # Max depth 3, 4, 10, 20

    # Split data into n train/test split folds
    kfolds = StratifiedKFold(n_splits=splits, shuffle=shuffle)

    # Create the model
    model = RandomForestClassifier()

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


def get_kmeans(x_train, y_train, x_test, y_true, xt_test):
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

    color_df['LAT'] = xt_test['Lat']
    color_df['LON'] = xt_test['Lng']

    print(color_df.head())

    all_blue = color_df[color_df['Color'] == 'blue']
    all_red = color_df[color_df['Color'] == 'red']

    plt.scatter(all_blue['LAT'], all_blue['LON'], color='blue', alpha=0.2)
    plt.scatter(all_red['LAT'], all_red['LON'], color='red', alpha=0.2)

    # plt.scatter(x_test, x_test, cmap={'blue', 1: 'red'})
    plt.show()\

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
    dir_path = 'D:/redwi/Documents/Thesis Data/'

    # Read in data from directories
    df = read_data(dir_path)

    # Fill empty values
    # df = clean_data(df)

    # Split into data and target values
    X_class = df['Reflectivity']            # Data for classification
    X_clust = df.drop(['Crop'], axis=1)     # Data for clustering
    y = df['Crop']

    # Encode target value
    y = encode_data(y)

    # Scale data
    scaler = StandardScaler()
    tmp = reshape_input_data(df['Reflectivity'])
    X_scaled = scaler.fit_transform(tmp)
    X_class = pd.DataFrame(X_scaled)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_class, y, test_size=.3, shuffle=True, stratify=y)

    # Reshape for models
    X_train = reshape_input_data(X_train)
    X_test = reshape_input_data(X_test)

    # Call models
    _ = get_decision_tree(X_train, Y_train, X_test, Y_test)

    _ = get_extra_tree(X_train, Y_train, X_test, Y_test, 'random')

    _ = get_random_forest(X_train, Y_train, X_test, Y_test)

# --------------------------------------------------------------------------------------------
    # Cross-Validation

    get_decision_tree_cv(X_class, y, splits=10)

    get_random_forest_cv(X_class, y, splits=10)

# --------------------------------------------------------------------------------------------
    # Course Grid Search

    get_random_forest_gs(X_train, Y_train, X_test, Y_test)

    # exit()

# --------------------------------------------------------------------------------------------
    # Clustering

    X_train_clust, X_test_clust, Y_train_clust, Y_test_clust = train_test_split(X_clust, y, test_size=.2, shuffle=True)

    x_train_clust = reshape_input_data(X_train_clust['Reflectivity'])
    x_test_clust = reshape_input_data(X_test_clust['Reflectivity'])

    get_kmeans(x_train_clust, Y_train, x_test_clust, Y_test, X_test_clust)

# -------------------------------------------------------------------------------------------
    # 'Deep' Learning Model



if __name__=="__main__":
    main()