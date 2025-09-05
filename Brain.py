from DStripper import DStripper # NOTE: this will be all 7 bands in all cases
from DataVis import DataVis
from Models import Class_Model, Regression_Model
import numpy as np
from scipy.spatial import cKDTree # this never looks like it imports correctly but it does
# NOTE: cKDTree, not ckdtree, even though auto-correct wants all lower case
import csv
from joblib import load
import math
from osgeo import gdal, osr

class Brain:
    def __init__(self):
        self.data_stripper = None
        self.truth_arr = None
        self.visualizer = DataVis()
        self.model = None # may be challenging to have one model variable to cover all possible models
        self.model_loaded = False

    def read_in_modis_data(self, hdf_path):
        self.data_stripper = DStripper(hdf_path)
        return True

    def create_training_data_from_box(self, output_path: str, lat_high, lat_low, lon_high, lon_low, classNum):
        band_box = self.data_stripper.strip_by_box(lat_high, lat_low, lon_high, lon_low)

        self.data_stripper.write_to_csv(output_path, band_box, classNum)

    def strip_from_boxes(self, boxes_csv_path: str, category, output_name: str):
        with open(boxes_csv_path) as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)

            def is_float(string):
                try:
                    float(string)
                    return True
                except:
                    return False

            i = 0
            while i < len(data) and len(data[i]) > 0:
                # if there is a digit in the line, gather lat/long point
                if is_float(data[i][0]):
                    max_lat = float(data[i][0])
                    min_lon = float(data[i][1])
                    min_lat = float(data[i + 1][0])
                    max_lon = float(data[i + 1][1])

                    result = self.data_stripper.strip_by_box(max_lat, min_lat, max_lon, min_lon)
                    self.data_stripper.write_to_csv(output_name, result, category)

                    i += 2 # this causes the loop to iterate by 2, so that no lines repeat

                else:
                    i += 1


    def read_in_truth_file(self, inputpath: str):
        # this will have to depend on read_in_truth_file

        # return [truth_val, lat, long] 3D array

        hdf_ds: gdal.Dataset = gdal.Open(inputpath)
        source_arr: np.ndarray = hdf_ds.ReadAsArray() # this assumes no subdatasets
        height, width = source_arr.shape

        # get transform and projection to determine lat/long
        geo_trans = hdf_ds.GetGeoTransform()
        proj = hdf_ds.GetProjection()

        # make spatial reference object which matches the dataset
        src = osr.SpatialReference()
        src.ImportFromWkt(proj)

        # make WGS84 spatial ref
        wgs = osr.SpatialReference()
        wgs.ImportFromEPSG(4326) # this results in WGS84

        # make object to convert source sinusoidal ref to wgs84
        transform = osr.CoordinateTransformation(src, wgs)

        vals_by_coord = np.zeros(shape=(height, width, 3), dtype=float) # these can be cast to int later if needed
        # this will hold the return vals

        # calculate lat/long per pixel
        def calc_transform(col: int, row: int, geo_trans, transform):
            # geo_trans : (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)
            x_proj = float(geo_trans[0] + col * geo_trans[1] + row * geo_trans[2])
            # origin_x + (pixel_number * pixel_width) + (line_number * rotation_x)
            y_proj = float(geo_trans[3] + float(col) * geo_trans[4] + float(row) * geo_trans[5])
            # origin_y + (pixel_number * rotation_y) + (line_number * pixel_height)

            # calculate transform to WGS84 and fill lons, lats
            lat, long, _ = transform.TransformPoint(x_proj, y_proj)
            return lat, long
        
        for i in range(height):
            for j in range(width):
                truth_val = source_arr[i][j]
                if math.isnan(truth_val):
                    truth_val = 0
                lat, long = calc_transform(i, j, geo_trans, transform)
                vals_by_coord[i][j][0] = truth_val
                vals_by_coord[i][j][1] = lat
                vals_by_coord[i][j][2] = long

        # vals_by_coord now done. Now save to disk
        self.truth_arr = vals_by_coord
        return vals_by_coord

    def write_truth_to_csv(self, input_path: str, output_path: str):
        # make 2D lat/long array from stripper to turn into kd tree
        # this is much faster than searching all values linearly
        if self.data_stripper is None:
            raise RuntimeError("No file loaded")
        
        latitudes = self.data_stripper.lats.ravel()
        longitudes = self.data_stripper.lons.ravel()

        coords = np.column_stack((latitudes, longitudes))

        bands_stack = np.stack([band.ravel() for band in self.data_stripper.bands], axis=1)

        # the lat/long for the pixel at bands_stack[k] are now at coords[k]

        tree = cKDTree(coords)
        # can now query tree for closest matching lat/long to get pixel values at that coordinate set

        # now call strip_truth_data to get truth data array
        if self.truth_arr is None:
            truth_array = self.read_in_truth_file(input_path) # read in truth data if it hasn't been recorded already
        else:
            truth_array = self.truth_arr

        with open(f"{output_path}", 'a', newline='') as file:
            writer = csv.writer(file)

            # this assumes NaN values were replaced with 0, as no other version has been implemented yet
            height, width, _ = truth_array.shape
            for i in range(height):
                for j in range(width):
                    lat = truth_array[i][j][1]
                    long = truth_array[i][j][2]

                    _, index = tree.query((lat, long), k=1) # find closest associated MODIS pixel
                    # NOTE: this uses pixel origin and may behave strangely if the box is outside the source MODIS data's dimensions

                    entry = [*bands_stack[index], *truth_array[i][j]]
                    writer.writerow(entry) # write all recorded bands, truth val, lat, long for pixel to file

        return True

    def train_model_csv(self, model_type: int, included_bands:list, useRatios: bool, training_data_path): # let type 0 be classification, 1 be regression
        if model_type == 0:
            self.model = Class_Model(included_bands, useRatios)
        else:
            self.model = Regression_Model(included_bands, useRatios)

        self.model.train_model(training_data_path)

        return self.model.get_stats(), self.model.get_importances()

    def write_predictions(self, results: np.ndarray, output_path: str):
        projection = self.data_stripper.get_projection()
        transform = self.data_stripper.get_transform()

        self.visualizer.write_predictions_tif_whole(results[0], f"{output_path}_predictions.tif", projection, transform)
        # have to split probs into something produceable

        if results[1] is not None: # will be None if regression model

            all_probs = results[1] # results[1] is a (2400 x 2400 x num_classes) ndarray

            # reshape probs from 2400x2400xNum_classes to Num_classesx2400x2400
            probs_arr = [all_probs[:,:,i] for i in range(all_probs.shape[-1])]

            for i in range(len(probs_arr)):
                self.visualizer.write_predictions_tif_whole(probs_arr[i], f"{output_path}_class{i}_probs.tif", projection, transform)
        return True
    
    def write_mean_filter_pred(self, predictions: np.ndarray, output_path: str):
        self.visualizer.write_mean_filter_predictions_tif_whole(
        predictions,  
        f"{output_path}_mean.tif", 
        self.data_stripper.get_projection(), 
        self.data_stripper.get_transform()
        )
        return True
    
    def load_old_model(self, model_path: str):
        self.model = load(model_path)
        return True
    
    def write_model(self, model_path: str):
        if model_path == '':
            raise Exception("No filename specified")
        return self.model.save_model_to_disk(model_path)
    
    def make_predictions(self):
        if self.model is None:
            raise Exception("No model loaded")
        
        # should work for either model type
        results = self.model.predict_for_whole_source(self.data_stripper.bands)

        return results
