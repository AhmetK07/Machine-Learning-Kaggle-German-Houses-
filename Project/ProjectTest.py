import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from keras.models import Sequential, load_model
from keras.layers import Dense
from math import sqrt

import os

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("germany_housing_data_14.07.2020.csv")

print("\n----------------------------------")
print(data.head())
print("\n----------------------------------")
print(data.info())
print("\n----------------------------------")
print(data.isnull().sum())
print("\n----------------------------------")
print()
print("Before labeling")

print(data['City'].value_counts())

print()
print("After labeling")

le = LabelEncoder()  # name.
data['City'] = le.fit_transform(data['City'])

print(data['City'].value_counts())
print("\n----------------------------------")

#print(le.classes_)

print("")
print("Before One Hot")
print("")
print(data['Condition'].value_counts())

one_hot = OneHotEncoder()
transformed_data = one_hot.fit_transform(data['Condition'].values.reshape(-1, 1)).toarray()

print("\n----------------------------------")
print(one_hot.categories_)

transformed_data = pd.DataFrame(transformed_data, columns=['modernized', 'refurbished', 'dilapidated', 'maintained',
                                                           'renovated', 'fixer-upper',
                                                           'first occupation after refurbishment',
                                                           'first occupation', 'by arrangement', 'as new', ''])
print("\n----------------------------------")
print(transformed_data.head())

###########################

print("Index value:")

print(transformed_data.iloc[90,])

print(data['Condition'][90])

print("\n----------------------------------")

print("After Deleting the Columns")

# Deleting Columns of the features which have about below %60 percent valid data.

del data["Energy_efficiency_class"]

del data["Energy_consumption"]

del data["Usable_area"]

del data["Year_renovated"]

print(data.isnull().sum())

print("\n----------------------------------")

print("After Deleting the Rows")

# Deleting Rows of the features which have about %95 percent valid data.

data = data.dropna(subset=['Heating'])

data = data.dropna(subset=['Type'])

data = data.dropna(subset=['Place'])

data = data.dropna(subset=['Year_built'])

data = data.dropna(subset=['Condition'])

data = data.dropna(subset=['Energy_certificate'])

print(data.isnull().sum())

print("\n----------------------------------")

print("After Imputation")

updated_data = data

updated_data['Energy_certificate_type'] = updated_data['Energy_certificate_type'].fillna("Unknown")

updated_data['Energy_source'] = updated_data['Energy_source'].fillna("Unknown")

updated_data['Bedrooms'] = updated_data['Bedrooms'].fillna(updated_data['Bedrooms'].mean())

updated_data['Bathrooms'] = updated_data['Bathrooms'].fillna(updated_data['Bathrooms'].mean())

updated_data['Floors'] = updated_data['Floors'].fillna(updated_data['Floors'].mean())

updated_data['Furnishing_quality'] = updated_data['Furnishing_quality'].fillna("Unknown")

updated_data['Garages'] = updated_data['Garages'].fillna(updated_data['Garages'].mean())

updated_data['Garagetype'] = updated_data['Garagetype'].fillna("Unknown")

updated_data['Free_of_Relation'] = updated_data['Free_of_Relation'].fillna("Unknown")

updated_data.info()

print("")
print("Updated version of the data and its null values.")

print(data.isnull().sum())

print("\n----------------------------------")

Space = data['Living_space'].values

Rooms = data['Rooms'].values

print("CORRELATION CALCULATION BETWEEN FEATURES:")

print(np.corrcoef(Space, Rooms))

print(pearsonr(Space, Rooms))
print("The correlation coefficient is 0.8112771883684731 and the two-tailed  p-value is 0.0 which means:")
print(
    "There is a direct correlation between these two variables because most of the time a good pearson value must be under 0.05")

print("\n----------------------------------")
print("")

print("Data head before MinMaxScaling")

HeadSpace = updated_data["Living_space"]

print("The whole datahead")
print(updated_data.head())

print("Specific Section Of Living_space")
print(HeadSpace.head())

scaler = MinMaxScaler()

updated_data['NormalizedSpace'] = scaler.fit_transform(data['Living_space'].values.reshape(-1, 1))

print("\n----------------------------------")
print("")
print("Standard Deviation & Mean || Min & Max in LivingSpace Before Min Max Scaling.")
print('means:', updated_data['Living_space'].values.mean(axis=0))
print('std:', updated_data['Living_space'].values.std(axis=0))

print('Min value:', updated_data['Living_space'].values.min(axis=0))
print('Max value:', updated_data['Living_space'].values.max(axis=0))

print("")
print("Standard Deviation & Mean || Min & Max in NormalizedSpace(Living_Space) Before Min Max Scaling.")

print('means:', updated_data['NormalizedSpace'].values.mean(axis=0))
print('std:', updated_data['NormalizedSpace'].values.std(axis=0))

print('Min value:', updated_data['NormalizedSpace'].values.min(axis=0))
print('Max value:', updated_data['NormalizedSpace'].values.max(axis=0))
print("\n----------------------------------")

print("")

print("Data head after MinMaxScaling")
UpdatedHeadSpace = updated_data["NormalizedSpace"]

print("The whole datahead")
print(updated_data.head())

print("Specific Living_Space converted to Normalized Space")
print(UpdatedHeadSpace.head())

print("\n----------------------------------")
print("")

# MY PREFERRED ALGORITHMS

# KNN Regression

TimeCalcStart = time.perf_counter()
print("KNN Regression Between Living Space & Garages ")

X = updated_data['Living_space'].values

Y = updated_data['Garages'].values

reg = KNeighborsRegressor()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

print("Our shape of the X_train: ", x_train.shape)

print("Our shape of the Y_train: ", y_train.shape)

knnreg = reg.fit(np.array(x_train).reshape(-1, 1), y_train)
pred = knnreg.predict(np.array(x_test.reshape(-1, 1)))

print("Our prediction: ", pred[0:5])

print("Our test values: ", y_test[0:5])

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")


# Fully Connected Neural Network

TimeCalcStart = time.perf_counter()

print("Fully Connected Neural Network: ")
print("")

def nnModel(x_train, y_train, epochs=200, batch_size=10):
    """
    A basic fully connected neural network model
    using relu activation function for hidden layers
    and sigmoid activation function for output layer

    parameters:
    x_train: training data
    x_test: testing data
    y_train: training labels
    y_test: testing labels
    epochs: number of epochs
    batch_size: batch size

    returns:
    model: model
    """
    if os.path.exists("model.h5"):
        return load_model("model.h5")

    model = Sequential()
    model.add(Dense(8, input_dim=4, activation="relu"))  # 1st  Hidden Layer input dim shows us the feature number
    model.add(Dense(16, activation="relu"))  # 2nd  Hidden Layer  # I've used relu for best values.
    model.add(Dense(1, activation="sigmoid"))  # 3rd is our output layer
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])  # I've used adam for perfomance purposes.
    # We are using binary_crossentropy and sigmoid just because our output values are binary (0 and 1).
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    model.save("model.h5")

    return model


X = [updated_data['Living_space'].values, updated_data['Bathrooms'].values, updated_data['Bedrooms'].values,
     updated_data['Floors'].values]

Y = updated_data['Price'].values


def normalize(updated_data):  # We are downgrading the values to the range of 0-1.
    tempArr = []

    for feature in updated_data:
        tempArr.append(stats.zscore(feature))
    print("Innormalize ", np.array(tempArr).shape)
    result = []

    for x in range(len(tempArr[0])):
        temp = []
        for y in range(len(tempArr)):
            temp.append(tempArr[y][x])

        result.append(temp)
    print("Innormalize ", np.array(result).shape)
    return result


# print(np.array(normalize(X)).shape)

X = normalize(X)

PriceMean = updated_data['Price'].values.mean(axis=0)

Y = [1 if value > PriceMean else 0 for value in Y]
print(len(X), len(Y))
print("\n----------------------------------")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

nn_model = nnModel(x_train, y_train) # Neural Network Model
print(nn_model.evaluate(x_test, y_test)) # Neural Network Model

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")


# Linear Regression

TimeCalcStart = time.perf_counter()

print("Linear Regression: ")
print("")

LinearReg = LinearRegression()

Y = [[value] for value in Y] # For changing the arrays to 2d array
model = LinearReg.fit(x_train, y_train)
pred = model.predict(x_test)

print("Prediction:", pred[0:5])
print("Actual Value:", y_test[0:5])

MeanSqErr = mean_squared_error(y_test, pred)

RMSqErr = sqrt(MeanSqErr)

print("Linear Regression rmse is:", RMSqErr)
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, pred)))

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")


# Random Forest

TimeCalcStart = time.perf_counter()

print("Random Forest: ")
print("")

Y = [[value] for value in Y]
RandFor = RandomForestRegressor(max_depth=10)
RandFor.fit(x_train, y_train)
pred = RandFor.predict(x_test)

print("Prediction: ", pred)

print("Actual value:", y_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, pred))

print('Mean Squared Error:', mean_squared_error(y_test, pred))

print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, pred)))

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")
# KNN Classification

TimeCalcStart = time.perf_counter()

Neigh = KNeighborsClassifier(n_neighbors=5)

Scaler = StandardScaler()

X = updated_data['Bedrooms'].values

Y = updated_data['Year_built'].values

X = [[value] for value in X]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

fit_scale = Scaler.fit(x_train)

x_train = Scaler.transform(x_train)

x_test = Scaler.transform(x_test)

model = Neigh.fit(x_train, y_train)

pred = model.predict(x_test)

print("KNN classification Prediction for key-mode: ", pred)
print("")

print("Actual Value: ", y_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, pred)))

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")

print("")
print("Random Selection: ")

TimeCalcStart = time.perf_counter()

ScoreN = Neigh.score(x_train, y_train)

Nmean = np.mean(ScoreN)

NStdDv = np.std(ScoreN)

print("Score: ", ScoreN)

print("Mean", Nmean)

print("Standard Deviation: ", NStdDv)

FiveF = cross_val_score(Neigh, x_train, y_train)

print("Five Fold calculation: ", FiveF)

TimeCalcEnd = time.perf_counter()

print("Total time: %f" % (TimeCalcEnd - TimeCalcStart))

print("\n----------------------------------")

