# include <pybind11/pybind11.h>
# include <pybind11/numpy.h>
# include <unordered_map>
# include <tuple>
# include <queue>


namespace py = pybind11;

struct Component {
    int id;
    int size;
    int start_row, start_col;
    int class_val;
    std::vector<std::pair<int, int>> pixels; // vector of pixel coordinates for all pixels in the components
};

bool check_in_bounds(int index, int limit) {
    return (index >= 0 && index < limit);
}

Component find_component_bfs(py::array_t<int> original, py::array_t<int> component_map, int start_row, int start_col, int component_id) {
    //find and store all pixels in a component, return the resulting component
    auto orig = original.unchecked<2>();
    auto comp_map = component_map.mutable_unchecked<2>();

    int width = original.shape(0);
    int height = original.shape(1);
    int match_class = orig(start_row, start_col);

    Component comp;
    comp.id = component_id;
    comp.size = 0;
    comp.start_row = start_row;
    comp.start_col = start_col;
    comp.class_val = match_class;

    std::queue<std::pair<int, int>> queue;
    queue.push({start_row, start_col});
    comp_map(start_row, start_col) = component_id;

    while (!queue.empty()) {
        auto [curr_row, curr_col] = queue.front();
        queue.pop();
        comp.pixels.push_back({curr_row, curr_col});
        comp.size++;

        // check adjacent pixels
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                int new_row = curr_row + i;
                int new_col = curr_col + j;

                if (check_in_bounds(new_row, width) && check_in_bounds(new_col, height) // if pixel is in bounds
                    && (comp_map(new_row, new_col) == -1) // and pixel not yet seen
                    && (orig(new_row, new_col) == match_class)) { // and class matches component
                        comp_map(new_row, new_col) = component_id; // mark as part of component
                        queue.push({new_row, new_col}); // add to queue to find adjecent pixels later
                }
            }
        }
    }
    return comp;
}

int find_most_common_neighbor(const Component& comp, py::array_t<int> original) {
    // find most common neighbor using pixels stored in comp
    auto orig = original.unchecked<2>();
    int width = original.shape(0);
    int height = original.shape(1);

    std::vector<int> neighbor_count(3,0); // holds counts of classes 0, 1, 2 for neighboring pixels
    std::vector<std::vector<bool>> counted(width, std::vector<bool>(height, false)); // make 2D boolean vector representing each pixel in the original image

    // for each pixel in the component, check its neighbors
    for (const auto& [row, col] : comp.pixels) {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }

                int new_row = row + i;
                int new_col = col + j;

                if (check_in_bounds(new_row, width) && check_in_bounds(new_col, height) // if pixel in bounds
                    && !counted[new_row][new_col]) { // and pixel not seen yet
                        int neighbor_class = orig(new_row, new_col);
                        if (neighbor_class != comp.class_val && // if neighbor class doesn't match
                            neighbor_class >= 0 && neighbor_class < 3) { // and neighbor class is in the expected trinary format
                                counted[new_row][new_col] = true;
                                neighbor_count[neighbor_class]++;
                            }
                    }
            }
        }
    }
    // determine most common neighbor
    int most_common = -1;
    int max_count = 0;
    for (int i = 0; i < 3; i++) {
        if (neighbor_count[i] > max_count) {
            most_common = i;
            max_count = neighbor_count[i];
        }
    }
    return most_common;
}

py::array_t<int> connected_component_filter_one_pass(py::array_t<int> predictions, int minSize) {
    int width = predictions.shape(0);
    int height = predictions.shape(1);

    // map each pixel to a component
    py::array_t<int> component_map = py::array_t<int>({width, height});
    auto comp_map = component_map.mutable_unchecked<2>();

    // initialize all values in component map to -1
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            comp_map(i, j) = -1;
        }
    }

    std::vector<Component> small_components;
    int component_id = 0;

    // single pass to determine all components
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (comp_map(i, j) == -1) { // if pixel not yet considered
                Component comp = find_component_bfs(predictions, component_map, i, j, component_id);

                if (comp.size < minSize) { // if component too small
                    small_components.push_back(std::move(comp)); // add to small components vector
                }
                component_id++;
            }
        }
    }

    // Process small components
    auto pred = predictions.mutable_unchecked<2>();
    for (const auto& comp: small_components) {
        int replacement_class = find_most_common_neighbor(comp, predictions);

        if (replacement_class != -1) {
            for (const auto& [row, col] : comp.pixels) {
                pred(row, col) = replacement_class;
            }
        }
    }

    return predictions;
}

py::array_t<int> majority_filter(py::array_t<int> predictions, int sensitivity) {
    auto pred = predictions.mutable_unchecked<2>();
    int width = predictions.shape(0);
    int height = predictions.shape(1);

    // create empty array to fill
    py::array_t<int> array = py::array_t<int>({width, height});
    auto arr = array.mutable_unchecked<2>();
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            arr(i, j) = static_cast<int>(0);
        }
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // for each pixel, create a map of seen class values
            std::unordered_map<int, int> seen = std::unordered_map<int, int>();
            // look at all neighboring pixels
            for (int row = -sensitivity; row < sensitivity + 1; row++) {
                for (int col = -sensitivity; col < sensitivity + 1; col++) {
                    // check if in bounds
                    if (check_in_bounds(i + row, width) && check_in_bounds(j + col, height)) {
                        int curr_pixel = pred(i + row, j + col);
                        // if class seen already, increase count
                        if (seen.count(curr_pixel) > 0) {
                            // increment
                            seen[curr_pixel] += 1;
                        } else {
                            // start count at 1
                            seen[curr_pixel] = 1;
                        }

                    }
                }
            }
            // determine most seen class val
            int most_seen = -1;
            int high_count = -1;
            for (auto [key, val] : seen) {
                if (val > high_count) {
                    most_seen = key;
                    high_count = val;
                }
            // set current pixel to most seen value
            }
            arr(i, j) = static_cast<int>(most_seen);
        }
    }
    return array;
}

py::array_t<int> land_majority_filter(py::array_t<int> predictions, int sensitivity) {
    auto pred = predictions.mutable_unchecked<2>();
    int width = predictions.shape(0);
    int height = predictions.shape(1);

    // create empty array to fill
    py::array_t<int> array = py::array_t<int>({width, height});
    auto arr = array.mutable_unchecked<2>();
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            arr(i, j) = static_cast<int>(0);
        }
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // for each pixel, create a map of seen class values
            std::unordered_map<int, int> seen = std::unordered_map<int, int>();
            // look at all neighboring pixels
            for (int row = -sensitivity; row < sensitivity + 1; row++) {
                for (int col = -sensitivity; col < sensitivity + 1; col++) {
                    // check if in bounds
                    if (check_in_bounds(i + row, width) && check_in_bounds(j + col, height)) {
                        int curr_pixel = pred(i + row, j + col);
                        // if class seen already, increase count
                        if (seen.count(curr_pixel) > 0) {
                            // increment
                            seen[curr_pixel] += 1;
                        } else {
                            // start count at 1
                            seen[curr_pixel] = 1;
                        }

                    }
                }
            }
            // determine most seen class val
            int most_seen = -1;
            int high_count = -1;
            for (auto [key, val] : seen) {
                if (val > high_count) {
                    most_seen = key;
                    high_count = val;
                }
            // set current pixel to most seen value
            }
            if (most_seen == 0) { // replace only if majority neighbor is land (value 0)
                arr(i, j) = static_cast<int>(most_seen);
            } else {
                arr(i, j) = static_cast<int>(pred(i, j));
            }
        }
    }
    return array;
}

py::array_t<float> mean_filter(py::array_t<float> predictions, int sensitivity) {
    auto pred = predictions.mutable_unchecked<2>();
    int width = predictions.shape(0);
    int height = predictions.shape(1);

    // create empty array to fill
    py::array_t<float> array = py::array_t<float>({width, height});
    auto arr = array.mutable_unchecked<2>();
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            arr(i, j) = static_cast<float>(0);
        }
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            // for each pixel, start a running sum of all adjacent values
            float sum = 0.0;
            float count = 0.0;
            // look at all neighboring pixels
            for (int row = -sensitivity; row < sensitivity + 1; row++) {
                for (int col = -sensitivity; col < sensitivity + 1; col++) {
                    // check if in bounds
                    if (check_in_bounds(i + row, width) && check_in_bounds(j + col, height)) {
                        float curr_pixel = pred(i + row, j + col);
                        count += 1.0;
                        sum += curr_pixel;
                    }
                }
            }
            // determine average pixel value
            float avg = sum / count;
            // set current pixel to most seen value
            arr(i, j) = static_cast<float>(avg);
        }
    }
    return array;
}

PYBIND11_MODULE(ImageProcessing, m) {
    m.doc() = "A set of raster image post processing functions for 500m resolution Algae detection.\nToo complex to be done quickly in Python";

    m.def("majority_filter", &majority_filter, "Replaces pixel values with the most common value of their neighbors.");
    m.def("land_majority_filter", &land_majority_filter, "Replaces pixel values with the most common neighbor only if that most common neighbor is land.");
    m.def("mean_filter", &mean_filter, "Replaces pixel values with the average of their neighbors.");
    m.def("connected_component_filter_one_pass", &connected_component_filter_one_pass, "A more efficient connected component filter.");

}