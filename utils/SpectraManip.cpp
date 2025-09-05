# include <pybind11/pybind11.h>
# include <pybind11/numpy.h>
#include <vector>

// namespace py = pybind11;

float calc_normalized_ratio(int num1, int num2) {
    return (num1 - num2) / (num1 + num2 + 0.000001);
}

float calc_normalized_ratio(double num1, double num2) {
    return (num1 - num2) / (num1 + num2 + 0.000001);
}

std::vector<std::vector<double>> calc_all_ratios(std::vector<std::vector<int>> spectra) {

    int num_pairs = int((spectra[0].size() * (spectra[0].size() - 1)) / 2);
    std::vector<std::vector<double>> ratios(int(spectra.size()), std::vector<double>(num_pairs, 0.0));

    // iterate through whole spectra array
    // for each row, calc ratio of each value to all values to the right
    // place these values in ratios vector

    for (int i = 0; i < int(spectra.size()); i++) { // i tracks row in spectra/ratios array
        int slotNum = 0; // tracks the position in the row for ratios array
        for (int j = 0; j < int(spectra[0].size()); j++) { // j tracks first number for ratio
            for (int k = j + 1; k < int(spectra[0].size()); k++) { // k tracks second number for ratio
                int num1 = spectra[i][j];
                int num2 = spectra[i][k];

                ratios[i][slotNum] = calc_normalized_ratio(num1, num2);
                slotNum += 1;
            }
        }
    }
    return ratios;
}

PYBIND11_MODULE(SpectraManip, m) {
    m.doc() = "A small toolset of functions to manipulate spectral information without relying on Python";
    
    m.def("calc_all_ratios", &calc_all_ratios, "Calculates normalized ratios for all values in dimension 1 of a 2D array. Requires ints in array and returns floating point values");
    m.def("calc_normalized_ratio", &calc_normalized_ratio, "Calculates the normalized ratio for two integers, returns a floating point number.");
}