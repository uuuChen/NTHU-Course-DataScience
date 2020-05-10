import pandas as pd
import numpy as np
import argparse

# data preprocess
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# evaluate
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("-trdp", "--train-data-path", type=str, default='train.csv')
parser.add_argument("-tedp", "--test-data-path", type=str, default='test.csv')
parser.add_argument("-ofp", "--output-file-path", type=str, default='output.csv')
args = parser.parse_args()


def load_data():
    df_train = pd.read_csv(args.train_data_path)
    df_test = pd.read_csv(args.test_data_path)
    df_test['RainToday'] = np.zeros((len(df_test)))
    len_of_train = len(df_train)
    df_train_test = pd.concat([df_train, df_test], sort=False)
    return df_train_test, len_of_train


def _set_bound(data, min_value, max_value):
    data = np.where(data > max_value, max_value, data)
    data = np.where(data < min_value, min_value, data)
    return data


def preprocess_data(df, len_of_train, drop_object=False):
    data_df, label_df = df.drop(columns=['RainToday']), df['RainToday']
    median_imr = SimpleImputer(missing_values=np.nan, strategy='median', copy=False)

    if drop_object:
        data_df = data_df.drop(columns=[col for col in df.columns if df[col].dtype == np.object])

    data_df_cols = data_df.columns
    for col in data_df_cols:
        flat_col = data_df[col].values.reshape(-1, 1)
        if data_df[col].dtypes == np.object:
            # fill dropped values of column by random value of column
            elements = list(set(data_df[col]))[1:]  # nan is in the first index
            data_df[col] = data_df[col].fillna(pd.Series(np.random.choice(elements, size=len(data_df[col].isnull()))))
            if col == 'Date':  # 'Date' has 3314 unique elements, so split them to year, month and day
                pass
            else:
                # convert to one hot label
                data_df = pd.concat([data_df, pd.get_dummies(data_df[col], prefix=col, drop_first=True)], axis=1)
                data_df = data_df.drop(col, axis=1)
        else:
            # fill dropped values of column by median value of column
            data_df[col] = median_imr.fit_transform(flat_col).ravel()

    # delete date column
    data_df = data_df.drop(['Date'], axis=1, errors='ignore')

    # get train, valid and test data and labels
    X_train, X_val, y_train, y_val = train_test_split(
        data_df.values[:len_of_train, :], label_df.values[:len_of_train], test_size=0.2, shuffle=True)
    X_test = data_df.values[len_of_train:, :]

    # fit normalizer before up-sampling
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train, y_train = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)

    # normalize
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(X_train.shape, y_train.shape)
    return (X_train, y_train), (X_val, y_val), X_test


def train_model(model, train_data_labels, val_data_labels):
    train_data, train_labels = train_data_labels
    val_data, val_labels = val_data_labels

    # build model
    model.fit(train_data, train_labels)

    # predict
    val_predicts = model.predict(val_data)

    # evaluate
    accuracy = accuracy_score(val_predicts, val_labels)
    f1_score_ = f1_score(val_predicts, val_labels)
    print('\n' + '-' * 70)
    print(f'model: {model}')
    print(f'Accuracy: {accuracy}')
    print(f'f1-score: {f1_score_}')

    return model, accuracy, f1_score_


def test_model(model, test_data):
    test_predicts = model.predict(test_data)
    return test_predicts


def write_csv(predicts):
    test_pred_df = pd.DataFrame(predicts.astype(int), columns=['RainToday'])
    test_pred_df.to_csv(args.output_file_path, index_label='Id')


def main():
    df_all, len_of_train = load_data()
    train_data_labels, val_data_labels, test_data = preprocess_data(df_all, len_of_train)

    # build models
    models = list()
    models.append(DecisionTreeClassifier())
    models.append(SVC(kernel='linear'))
    models.append(SVC(kernel='poly'))
    models.append(RandomForestClassifier(n_estimators=100))
    models.append(KMeans(n_clusters=2))
    models.append(KNeighborsClassifier(2))
    models.append(SVC(kernel='rbf'))
    models.append(AdaBoostClassifier())
    models.append(GaussianNB())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())
    models.append(LogisticRegression(solver='liblinear', random_state=0))
    models.append(XGBClassifier())

    # train models and print results
    best_f1_score = 0
    best_model = None
    for model in models:
        model, accuracy, f1_score_ = train_model(model, train_data_labels, val_data_labels)
        if f1_score_ > best_f1_score:
            best_f1_score = f1_score_
            best_model = model

    # test model
    test_predicts = test_model(best_model, test_data)
    write_csv(test_predicts)


if __name__ == '__main__':
    main()


