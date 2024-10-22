import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the data
def the_housing_data():
    return pd.read_csv("housing.csv")

def get_first_five_entries_of(data):
    return data.head()

housing_districts = the_housing_data()

# Preform stratified sampling to guarantee proportianal representation of classes between train and test sets
def get_income_categories_for(data):
    return pd.cut(data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

def get_income_category_histogram_for(data):
    return data["income_cat"].hist()

def get_data_as_train_and_test_by_income_category(data):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
        return [strat_train_set, strat_test_set]

def get_distribution_of_income_categories_in_test_set(test_set):
    test_set["income_cat"].value_counts() / len(test_set)

def remove_income_category_from_train_and_test_sets(train_set, test_set):
    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

def get_copy_of_set(set):
    return set.copy()

housing_districts["income_cat"] = get_income_categories_for(housing_districts)
get_income_category_histogram_for(housing_districts)

train_and_test_sets = get_data_as_train_and_test_by_income_category(housing_districts)
train_set = train_and_test_sets[0]
test_set = train_and_test_sets[1]
get_distribution_of_income_categories_in_test_set(test_set)
remove_income_category_from_train_and_test_sets(train_set, test_set)

housing_districts = train_set.copy()

# Backfill missing numerical data with the median value for each filter
def get_features_and_labels_for_train_set(train_set):
    feature_data = train_set.drop("median_house_value", axis = 1)
    labels = train_set["median_house_value"].copy()
    return [feature_data, labels]

def get_imputer():
    return SimpleImputer(strategy="median")

def get_numerical_data_with_backfilled_missing_values_for(data, imputer):
    numerical_data = data.drop("ocean_proximity", axis=1)
    imputer.fit(numerical_data)
    print('Values used to backfill missing data:', imputer.statistics_)
    return numerical_data

def print_median_values_for_each_feature_within(numerical_data):
    print('Median values for each feature:', numerical_data.median().values)

def transform_numerical_data_with_imputer(numerical_data, imputer):
    return imputer.transform(numerical_data)

imputer = get_imputer()

feature_data_and_labels = get_features_and_labels_for_train_set(train_set)
housing_districts = feature_data_and_labels[0]
housing_district_labels = feature_data_and_labels[1]

housing_districts_numerical_data = get_numerical_data_with_backfilled_missing_values_for(housing_districts, imputer)

# Prepare the housing district data
rooms_column_index, bedrooms_column_index, population_column_index, households_column_index = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
            
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_column_index] / X[:, households_column_index]
        population_per_household = X[:, population_column_index] / X[:, households_column_index]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, population_column_index] / X[:, rooms_column_index]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def get_numerical_transformation_pipeline(data):
    combined_attribute_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    extra_attributes = combined_attribute_adder.transform(data.values)
    return Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('combined_attribute_adder', combined_attribute_adder),
        ('std_scaler', StandardScaler())
    ])

def get_categorical_encoder():
    return OneHotEncoder()

numerical_attributes = list(housing_districts_numerical_data)
categorical_attributes = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", get_numerical_transformation_pipeline(housing_districts), numerical_attributes),
    ("cat", get_categorical_encoder(), categorical_attributes)
])

housing_districts_prepared = full_pipeline.fit_transform(housing_districts)

# Implement the first model
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_districts_prepared, housing_district_labels)

housing_district_median_price_predictions = forest_reg.predict(housing_districts_prepared)

# Evaluate the first model's preformance
forest_mse = mean_squared_error(housing_district_labels, housing_district_median_price_predictions)
forest_rmse = np.sqrt(forest_mse)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def get_rmse_scores_for_random_forest_regressor(forest_reg):
    forest_scores = cross_val_score(forest_reg, housing_districts_prepared, housing_district_labels, scoring = "neg_mean_squared_error", cv = 10)
    return np.sqrt(-forest_scores)

display_scores(get_rmse_scores_for_random_forest_regressor(forest_reg))

# Identify ideal hyperparameter values
def get_ideal_hyperparameters_with_grid_search(feature_data, labels):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
    forest_reg = RandomForestRegressor()
    
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(feature_data, labels)
    print(grid_search.best_params_)
    return grid_search

grid_search = get_ideal_hyperparameters_with_grid_search(housing_districts_prepared, housing_district_labels)

# Create the final model and get the housing district median price predictions for the test data
def get_prepared_data_for(data):
    return full_pipeline.transform(data)

def get_housing_district_median_price_predictions_for(data, model):
    prepared_data = get_prepared_data_for(data)
    return model.predict(prepared_data)

test_housing_districts = test_set.drop("median_house_value", axis = 1)
test_housing_district_labels = test_set["median_house_value"].copy()

final_model = grid_search.best_estimator_

housing_district_median_price_predictions = get_housing_district_median_price_predictions_for(test_housing_districts, final_model)

# Evaluate the final model's preformance
def print_root_mean_square_error(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    print("Average estimation is off by: $", rmse)

print_root_mean_square_error(test_housing_district_labels, housing_district_median_price_predictions)

joblib.dump(final_model, "housing_district_median_price_predictor.pkl")