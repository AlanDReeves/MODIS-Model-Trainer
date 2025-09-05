import numpy as np
from osgeo import gdal, osr
import csv
from affine import Affine

class DStripper:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.bands: np.ndarray = None
        self.lons: np.ndarray = None
        self.lats: np.ndarray = None
        self.category: np.ndarray = None
        self.dpoints: np.ndarray = None

        gdal.UseExceptions()

        self.strip_from_hdf()
        self.find_lat_long()
        self.make_DPoints()

    def strip_from_hdf(self): # This function is hardcoded to only work correctly on MYD09GA files
        bands = np.zeros(shape=(7, 2400, 2400), dtype=int)
        for i in range(1, 8):
            # check if each set can be opened
            add_set:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b0{i}_1')
            if add_set is None:
                # if not opened, raise exception
                raise RuntimeError(f"Band {i} Failed to Open")
            else:
                # append as ndArray
                # bands.append(add_set.ReadAsArray())
                bands[i - 1] = add_set.ReadAsArray()

        self.bands = bands
        return bands
    
    def find_lat_long(self):
        # open an arbitrary set to gather transform and projection data
        temp_set: gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')

        geo_trans = temp_set.GetGeoTransform() # (origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height)
        proj = temp_set.GetProjection() # gets a string that describes the spatial reference system

        # make a spatial reference object which matches the dataset loaded (sinusoidal)
        src = osr.SpatialReference()
        src.ImportFromWkt(proj)

        # make a spatial reference object with WGS84 reference system
        dst = osr.SpatialReference()
        dst.ImportFromEPSG(4326)

        # make an object which converts sinusoidal reference to WGS84
        transform = osr.CoordinateTransformation(src, dst)

        temp_band_ref = self.bands[0]

        self.lons = np.zeros(temp_band_ref.shape)
        self.lats = np.zeros(temp_band_ref.shape)

        def calc_transform(self, col: int, row: int, geo_trans, transform):
                x_proj = geo_trans[0] + float(col) * geo_trans[1] + float(row) * geo_trans[2]
                # origin_x + (pixel_number * pixel_width) + (line_number * rotation_x)
                y_proj = float(geo_trans[3] + float(col) * geo_trans[4] + float(row) * geo_trans[5])
                # origin_y + (pixel_number * rotation_y) + (line_number * pixel_height)

                # calculate transform to WGS84 and fill lons, lats
                lat, lon, _ = transform.TransformPoint(x_proj, y_proj)
                self.lons[col, row] = lon
                self.lats[col, row] = lat


        for col in range(len(temp_band_ref)):
            for row in range(len(temp_band_ref[0])):
                calc_transform(self, col, row, geo_trans, transform)

        return True

    def make_DPoints(self):
        dpoints = np.stack([
            self.bands[0], self.bands[1], 
            self.bands[2], self.bands[3],
            self.bands[4], self.bands[5],
            self.bands[6], 
            self.lats, self.lons], axis=-1)
        # creates 2400x2400x9 ndArray
        self.dPoints = dpoints
        return True
    
    def strip_by_box(self, lat_high: float, lat_low: float, long_high: float, long_low: float):
        result_list = []
        for col in range(len(self.dPoints)):
            # create new col to add to list
            new_col = []
            for row in range(len(self.dPoints[0])):
                dp = self.dPoints[col, row]
                lat_good = False
                long_good = False
                # check if location within lat/long box
                # if inside limits, add band info, lat/long, isTarget to array
                if lat_low <= dp[7] <= lat_high:
                    lat_good = True
                if long_low <= dp[8] <= long_high:
                    long_good = True

                if lat_good and long_good:
                    new_col.append(dp)
            if len(new_col) > 0:
                result_list.append(new_col)

        # returns list because it needs to have variable dimensions and np arrays cannot do that
        return result_list
    
    def get_transform(self):
        dSet:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')
        transform = dSet.GetGeoTransform()
        transform = Affine.from_gdal(*transform)
        return transform

    def get_projection(self):
        dSet:gdal.Dataset = gdal.Open(f'HDF4_EOS:EOS_Grid:"{self.file_path}":MODIS_Grid_500m_2D:sur_refl_b01_1')
        return dSet.GetProjection()
    
    def write_to_csv(self, output_filename: str, box_data: list, category: int):
        """box_data: must be a 3D list, with the 3rd axis being the bands and lat/long data"""
        with open(output_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for col in box_data:
                for row in col:
                    file_row = [
                        int(row[0]), 
                        int(row[1]), 
                        int(row[2]), 
                        int(row[3]), 
                        int(row[4]),
                        int(row[5]),
                        int(row[6]),
                        int(category), 
                        float(row[7]), 
                        float(row[8])
                        ]
                    writer.writerow(file_row)
        return True
            
