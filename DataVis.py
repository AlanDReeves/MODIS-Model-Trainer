import numpy as np
import rasterio
from affine import Affine
from utils import ImageProcessing

class DataVis:

    def write_tif(self, dpoints: np.ndarray, output_path: str, projection: str, transform: Affine):
        """Writes bands 1-7 to a tif at the path indicated as param output_path"""
        # gather data from dpoints
        rows, cols, _ = dpoints.shape

        band1 = np.zeros((rows, cols), dtype=np.int16)
        band2 = np.zeros((rows, cols), dtype=np.int16)
        band3 = np.zeros((rows, cols), dtype=np.int16)
        band4 = np.zeros((rows, cols), dtype=np.int16)
        band5 = np.zeros((rows, cols), dtype=np.int16)
        band6 = np.zeros((rows, cols), dtype=np.int16)
        band7 = np.zeros((rows, cols), dtype=np.int16)

        for i in range(cols):
            for j in range(rows):
                band1[i, j] = (dpoints[i][j][0])
                band2[i, j] = (dpoints[i][j][1])
                band3[i, j] = (dpoints[i][j][2])
                band4[i, j] = (dpoints[i][j][3])
                band5[i, j] = (dpoints[i][j][4])
                band6[i, j] = (dpoints[i][j][5])
                band7[i, j] = (dpoints[i][j][6])

        # write to tif
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=4,
            dtype=np.int16,
            crs=projection,
            transform=transform
        ) as dst:
            dst.write(band1, 1)
            dst.write(band2, 2)
            dst.write(band3, 3)
            dst.write(band4, 4)
            dst.write(band5, 5)
            dst.write(band6, 6)
            dst.write(band7, 7)

    def write_predictions_tif_whole(self, predictions: np.ndarray[int], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform\n
        Can handle floats, despite typing in params"""
        rows, cols = predictions.shape

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            # add a fake dimension so that rastorio isn't confused.
            dst.write(predictions[np.newaxis, :, :])

    def write_majority_filter_predictions_tif_whole(self, predictions: np.ndarray[int], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = predictions.shape

        processed_pred = ImageProcessing.majority_filter(predictions, 2)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            # add a fake dimension so that rastorio isn't confused.
            dst.write(processed_pred[np.newaxis, :, :])

    def write_mean_filter_predictions_tif_whole(self, predictions: np.ndarray[float], output_path: str, projection: str, transform: Affine):
            """Writes binary predictions to a tif file matching the source projection and transform\n
            Requires the prediction array to match the size of the original transform"""
            rows, cols = predictions.shape

            processed_pred = ImageProcessing.mean_filter(predictions, 1)

            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=rows,
                width=cols,
                count=1,
                dtype="float32",
                crs=projection,
                transform=transform,
            ) as dst:
                # add a fake dimension so that rastorio isn't confused.
                dst.write(processed_pred[np.newaxis, :, :])

    def write_land_majority_filter_predictions_tif_whole(self, predictions: np.ndarray[int], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = predictions.shape

        processed_pred = ImageProcessing.land_majority_filter(predictions, 2)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            # add a fake dimension so that rastorio isn't confused.
            dst.write(processed_pred[np.newaxis, :, :])

    def write_connected_filter_predictions_tif_whole(self, predictions: np.ndarray[int], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = predictions.shape

        processed_pred = ImageProcessing.connected_component_filter_one_pass(predictions, 10)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            # add a fake dimension so that rastorio isn't confused.
            dst.write(processed_pred[np.newaxis, :, :])

    def write_probs_tif_whole(self, probs: np.ndarray[float], output_path: str, projection: str, transform: Affine):
        """Writes binary predictions to a tif file matching the source projection and transform\n
        Requires the prediction array to match the size of the original transform"""
        rows, cols = probs.shape

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=projection,
            transform=transform,
        ) as dst:
            dst.write(probs[np.newaxis, :, :])

    def get_col_row(self, dpoints: np.ndarray, lon: float, lat: float):
        min_dist = float('inf')
        best_col, best_row = -1, 1

        for col in range(len(dpoints)):
            for row in range(len(dpoints[0])):
                dp = dpoints[col][row]
                dist = abs(dp[4] - lat) + abs(dp[5] - lon)

                if dist < min_dist:
                    min_dist = dist
                    best_col, best_row = col, row
        
        if best_col == -1 or best_row == -1:
            raise ValueError("No matching point found for given lat/lon.")
        
        return best_col, best_row
                
    def reform_box(self, lat_high: float, lat_low: float, long_high: float, long_low: float, dpoints: np.ndarray):
        """[start_col, end_col, start_row, end_row]"""
        col1, row1 = self.get_col_row(dpoints, long_high, lat_high)
        col2, row2 = self.get_col_row(dpoints, long_low, lat_low)

        start_col = min(col1, col2)
        end_col = max(col1, col2)
        start_row = min(row1, row2)
        end_row = max(row1, row2)

        return [start_col, end_col, start_row, end_row]