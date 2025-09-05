import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from joblib import dump

class Class_Model:
    def __init__(self, included_bands: list, useRatios: bool):
        self.model: RandomForestClassifier = None
        self.global_Imputer: SimpleImputer = None
        self.spectra_train = None
        self.spectra_test = None
        self.results_train = None
        self.results_test = None
        self.included_bands = included_bands
        self.useRatios = useRatios

    def calc_normalized_ratio(self, num1, num2):
        return (num1 - num2) / (num1 + num2 + 1e-6)
    
    def calc_all_ratios(self, spectra):
        # calculate the number of ratios created by however many bands were selected
        # sectra is a 2D array
        num_pairs = int((len(spectra[0]) * (len(spectra[0]) - 1)) / 2)
        ratios = np.zeros(shape=(len(spectra), num_pairs), dtype=float)

        # calculate ratios for every band provided

        # cycle through selected bands
        # compare every band to those to its right
        for k in range(len(spectra)): # k tracks which data point is being considered
            slot_num = 0 # tracks which ratio is currently being calculated for the given data point
            row = np.zeros(shape=(num_pairs), dtype=float)
            for i in range(len(spectra[0])):
                for j in range(i + 1, len(spectra[0])):
                    num1 = spectra[k][i]
                    num2 = spectra[k][j]

                    ratio = self.calc_normalized_ratio(num1, num2)
                    row[slot_num] = ratio
                    slot_num += 1
            ratios[k] = row # insert completed row of calculated ratios into list
        return ratios
    
    def train_model(self, training_data_path: str):
        # read in data
        with open(training_data_path) as file:
            csv_data = csv.reader(file)
            # convert to list
            csv_data = list(csv_data)
            # remove header if there is one
            if not any(char.isdigit() for char in csv_data[0]):
                csv_data = csv_data[1:]

        # separate spectral data and results
        # it's fine to close the file now
        spectra: np.array = np.zeros((len(csv_data), 7), dtype=int)
        results: np.array = np.zeros((len(csv_data)), dtype=int) # represents the "class" of the pixel i.e. algae, clean water, etc
        # fill spectra and results
        # [band1, band2, band3, band4, band5, band6, band7, result, lat, long]
        for i in range(len(csv_data)):
            for j in range(7):
                spectra[i][j] = int(csv_data[i][j])

            results[i] = int(csv_data[i][7])

        # replace bad data with np.nan. 
        # see MOD09 user guide for details
        spectra = np.where(spectra < -100, np.nan, spectra)

        # create mask to show where values were bad
        # this allows missingness to be accounted for in predictions
        missing_mask = np.isnan(spectra)[:, self.included_bands]

        # impute both targets and non-targets together
        # imputer will not be usable for predictions otherwise
        self.global_Imputer = SimpleImputer(strategy='mean')
        spectra_imputed = self.global_Imputer.fit_transform(spectra)

        final_vals: np.ndarray = None

        # calculate ratios if asked for
        if self.useRatios:
            # cut to only desired bands and pass
            spectra_imputed = spectra_imputed[:, self.included_bands]
            final_vals = self.calc_all_ratios(spectra_imputed) # this calculates ratios only for desired bands

        else:
            # use only the desired bands
            num_bands = len(self.included_bands)
            cut_bands = np.zeros(shape=(len(spectra_imputed), num_bands), dtype=int)
            row = np.zeros(shape=(num_bands), dtype=int)

            for k in range(len(spectra_imputed)):
                for i in range(len(row)):
                    row[i] = spectra_imputed[k][self.included_bands[i]]
                cut_bands[k] = row
            
            final_vals = cut_bands

        # add on missing_mask
        spectra_finished = np.hstack([final_vals, missing_mask.astype(int)])

        # split into train and test data
        self.spectra_train, self.spectra_test, self.results_train, self.results_test = train_test_split(spectra_finished, results, test_size=0.2)

        # now train model
        self.model = RandomForestClassifier()
        self.model.fit(self.spectra_train, self.results_train)

    def get_stats(self):
        # test model on remaining test data
        if self.model is None:
            return None
        results_prediction = self.model.predict(self.spectra_test)

        accuracy = accuracy_score(self.results_test, results_prediction)
        result = f"Accuracy: {accuracy: .2f}\nConfusion Matrx:\n{confusion_matrix(self.results_test, results_prediction)}\nClassification Report: \n{classification_report(self.results_test, results_prediction)}"
        return result
    
    def get_importances(self):
        if self.model is None:
            return None
        
        importances = self.model.feature_importances_
        feature_names = []

        if self.useRatios:
            for i in range(len(self.included_bands)):
                for j in range(i + 1, len(self.included_bands)):
                    feature_names.append(f"band {self.included_bands[i] + 1} / band {self.included_bands[j] + 1}")

        else:
            for i in range(len(self.included_bands)):
                feature_names.append(f"band {self.included_bands[i] + 1}")

        for i in range(len(self.included_bands)):
            feature_names.append(f"band {self.included_bands[i] + 1} missingness")

        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        return importances_df
    
    def predict_for_whole_source(self, bands: np.ndarray):
        height, width = bands[0].shape # bands should be (7 x 2400 x 2400)
        # create a 2D vector representing all pixels. This is necessary to pass to the scikit learn model
        twoD_bands = bands.reshape(7, -1).T # this reshapes to 7 x 5,670,000 and transposes to 5,670,000 x 7

        # impute
        twoD_bands_imputed = self.global_Imputer.transform(twoD_bands)

        # cut down to only desired bands
        twoD_bands_imputed = twoD_bands_imputed[:, self.included_bands]

        # create missing_mask
        missing_mask = np.isnan(twoD_bands)[:, self.included_bands]

        final_vals: np.ndarray

        if self.useRatios:
            final_vals = self.calc_all_ratios(twoD_bands_imputed)
        else:
            final_vals = twoD_bands_imputed

        spectra_finished = np.hstack((final_vals, missing_mask))

        predictions = self.model.predict(spectra_finished)
        probabilities = self.model.predict_proba(spectra_finished)

        predictions = predictions.reshape(height, width)
        probabilities = probabilities.reshape(height, width, -1)

        return predictions, probabilities # NOTE: probabilities will still contain the probabilities of each class

    def save_model_to_disk(self, file_path):
        dump(self, file_path)
        return True


class Regression_Model:
    def __init__(self, included_bands: list, useRatios: bool):
        self.model: RandomForestRegressor = None
        self.global_Imputer: SimpleImputer = None
        self.spectra_train = None
        self.spectra_test = None
        self.results_train = None
        self.results_test = None
        self.included_bands = included_bands
        self.useRatios = useRatios

    def calc_normalized_ratio(self, num1, num2):
        return (num1 - num2) / (num1 + num2 + 1e-6)
    
    def calc_all_ratios(self, spectra):
        # calculate the number of ratios created by however many bands were selected
        # sectra is a 2D array
        num_pairs = int((len(spectra[0]) * (len(spectra[0]) - 1)) / 2)
        ratios = np.zeros(shape=(len(spectra), num_pairs), dtype=float)

        # calculate ratios for every band provided

        # cycle through selected bands
        # compare every band to those to its right
        for k in range(len(spectra)): # k tracks which data point is being considered
            slot_num = 0 # tracks which ratio is currently being calculated for the given data point
            row = np.zeros(shape=(num_pairs), dtype=float)
            for i in range(len(spectra[0])):
                for j in range(i + 1, len(spectra[0])):
                    num1 = spectra[k][i]
                    num2 = spectra[k][j]

                    ratio = self.calc_normalized_ratio(num1, num2)
                    row[slot_num] = ratio
                    slot_num += 1
            ratios[k] = row # insert completed row of calculated ratios into list
        return ratios
    
    def train_model(self, training_data_path: str):
        # read in data
        with open(training_data_path) as file:
            csv_data = csv.reader(file)
            # convert to list
            csv_data = list(csv_data)
            # remove header if there is one
            if not any(char.isdigit() for char in csv_data[0]):
                csv_data = csv_data[1:]

        # separate spectral data and results
        # it's fine to close the file now
        spectra: np.array = np.zeros((len(csv_data), 7), dtype=int)
        results: np.array = np.zeros((len(csv_data)), dtype=float) # represents the value of the pixel which the model will predict later
        # fill spectra and results
        # [band1, band2, band3, band4, band5, band6, band7, result, lat, long]
        for i in range(len(csv_data)):
            for j in range(7):
                spectra[i][j] = int(csv_data[i][j])

            results[i] = float(csv_data[i][7])

        # replace bad data with np.nan. 
        # see MOD09 user guide for details
        spectra = np.where(spectra < -100, np.nan, spectra)

        # create mask to show where values were bad
        # this allows missingness to be accounted for in predictions
        missing_mask = np.isnan(spectra)[:, self.included_bands]

        # impute both targets and non-targets together
        # imputer will not be usable for predictions otherwise
        self.global_Imputer = SimpleImputer(strategy='mean')
        spectra_imputed = self.global_Imputer.fit_transform(spectra)

        final_vals: np.ndarray = None

        # calculate ratios if asked for
        if self.useRatios:
            # cut to only desired bands and pass
            spectra_imputed = spectra_imputed[:, self.included_bands]
            final_vals = self.calc_all_ratios(spectra_imputed) # this calculates ratios only for desired bands

        else:
            # use only the desired bands
            num_bands = len(self.included_bands)
            cut_bands = np.zeros(shape=(len(spectra_imputed), num_bands), dtype=int)
            row = np.zeros(shape=(num_bands), dtype=int)

            for k in range(len(spectra_imputed)):
                for i in range(len(row)):
                    row[i] = spectra_imputed[k][self.included_bands[i]]
                cut_bands[k] = row
            
            final_vals = cut_bands

        # add on missing_mask
        spectra_finished = np.hstack([final_vals, missing_mask.astype(int)])

        # split into train and test data
        self.spectra_train, self.spectra_test, self.results_train, self.results_test = train_test_split(spectra_finished, results, test_size=0.2)

        # now train model
        self.model = RandomForestRegressor()
        self.model.fit(self.spectra_train, self.results_train)

    def get_stats(self):
        # test model on remaining test data
        if self.model is None:
            return None
        results_prediction = self.model.predict(self.spectra_test)
        mse = mean_squared_error(self.results_test, results_prediction)
        rmse = root_mean_squared_error(self.results_test, results_prediction)
        r2 = r2_score(self.results_test, results_prediction)

        result = [
            f"Mean Squared Error: {mse}",
            f"Root Mean Squared Error: {rmse}",
            f"R2 Score: {r2}"
        ]
        
        return result
    
    def get_importances(self):
        if self.model is None:
            return None
        
        importances = self.model.feature_importances_
        feature_names = []

        if self.useRatios:
            for i in range(len(self.included_bands)):
                for j in range(i + 1, len(self.included_bands)):
                    feature_names.append(f"band {self.included_bands[i] + 1} / band {self.included_bands[j] + 1}")

        else:
            for i in range(len(self.included_bands)):
                feature_names.append(f"band {self.included_bands[i] + 1}")

        for i in range(len(self.included_bands)):
            feature_names.append(f"band {self.included_bands[i] + 1} missingness")

        importances_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False)

        return importances_df
    
    def predict_for_whole_source(self, bands: np.ndarray):
        height, width = bands[0].shape # bands should be (7 x 2400 x 2400)
        # create a 2D vector representing all pixels. This is necessary to pass to the scikit learn model
        twoD_bands = bands.reshape(7, -1).T # this reshapes to 7 x 5,670,000 and transposes to 5,670,000 x 7

        # impute
        twoD_bands_imputed = self.global_Imputer.transform(twoD_bands)

        # cut down to only desired bands
        twoD_bands_imputed = twoD_bands_imputed[:, self.included_bands]

        # create missing_mask
        missing_mask = np.isnan(twoD_bands)[:, self.included_bands]

        final_vals: np.ndarray

        if self.useRatios:
            final_vals = self.calc_all_ratios(twoD_bands_imputed)
        else:
            final_vals = twoD_bands_imputed

        spectra_finished = np.hstack((final_vals, missing_mask))

        predictions = self.model.predict(spectra_finished)
        # probability prediction does not exist for regressor

        predictions = predictions.reshape(height, width)

        return predictions, None # higher level function calls require a tuple

    def save_model_to_disk(self, file_path):
        dump(self, file_path)
        return True