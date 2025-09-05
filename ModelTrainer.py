# This serves as the controller
from GUI import GenGUI
from tkinter import filedialog, simpledialog
from Brain import Brain
import threading
import os
from utils import popup_calls

gui = GenGUI()

cur_hdf = None
cur_doc = None

brain = Brain()

def check_thread_progress(thread: threading.Thread):
    if thread.is_alive():
        gui.root.after(0, lambda: gui.write_to_terminal_same_line('.')) # write a dot if thread is still working
        gui.root.after(500, lambda: check_thread_progress(thread))
    else:
        gui.root.after(0, lambda: gui.write_to_terminal("")) # else write blank message to get new line


def read_in_modis():
    cur_hdf = filedialog.askopenfilename(defaultextension=".hdf", title="Select the MYD09GA file to open")
    gui.write_to_terminal("Opening file and reading. This may take a minute")

    def worker():
        try:
            brain.read_in_modis_data(cur_hdf)
            modis_name = os.path.basename(cur_hdf)

            gui.root.after(0, lambda: gui.write_to_terminal("Complete"))
            gui.root.after(0, lambda: gui.MODIS_file_label.config(text=f"MODIS file: {modis_name}"))
        except Exception as e:
            gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Failed to open file: {e}"))
    
    thread1 = threading.Thread(target=worker, daemon=True)
    thread1.start()
    check_thread_progress(thread1)


def read_in_truth():
    cur_doc = filedialog.askopenfilename(defaultextension=".hdf", title="Select the truth hdf file to open")
    gui.root.after(0, lambda: gui.write_to_terminal("Reading in truth file"))

    def worker():
        try:
            brain.read_in_truth_file(cur_doc)
            doc_name = os.path.basename(cur_doc)

            gui.root.after(0, lambda: gui.write_to_terminal("Complete"))
            gui.root.after(0, lambda: gui.DOC_file_label.config(text=f"Truth file: {doc_name}"))
        except Exception as e:
            gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Failed to open file: {e}"))

    thread1 = threading.Thread(target=worker, daemon=True)
    thread1.start()
    check_thread_progress(thread1)


def train_model_csv():
    model_type = simpledialog.askinteger(title="Model Type?", prompt="Use Classification model (0) or Regression model (1)?", parent=gui.root)
    
    included_bands = popup_calls.select_bands(gui.root, 7)

    useRatios = simpledialog.askinteger(title="Ratios or raw data?", prompt="Enter 1 for normalized ratios, 0 for raw data", initialvalue=1, minvalue=0, maxvalue=1, parent=gui.root)
    useRatios = bool(useRatios)
    training_data = filedialog.askopenfilename(defaultextension=".csv", title="Select the training data file to open")
    gui.root.after(0, lambda: gui.write_to_terminal("Starting training. This may take a long time for large file sizes"))

    def worker():
        try:
            stats, importances = brain.train_model_csv(model_type, included_bands, useRatios, training_data)
            gui.root.after(0, lambda:gui.write_to_terminal(""))# empty line for spacing
            gui.root.after(1, lambda:gui.write_to_terminal_same_line(stats)) # same line version to avoid concat exception
            gui.root.after(2, lambda:gui.write_to_terminal("")) # empty line for spacing
            gui.root.after(3, lambda:gui.write_to_terminal_same_line(importances)) #same line version to avoid concat exception
            gui.root.after(4, lambda:gui.write_to_terminal(""))
        except Exception as e:
            gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to train model: {e}"))

    thread1 = threading.Thread(target=worker, daemon=True)
    thread1.start()
    check_thread_progress(thread1)


def create_training_data():
    # determine if from box or from truth
    thread1: threading.Thread
    type_indicator = simpledialog.askinteger(title="Method selection", prompt="Gather training data from box (0) or from truth data? (1)"
                            , initialvalue=0, maxvalue=1, minvalue=0, parent=gui.root)
    if type_indicator == 0:
        # box
        strip_by_box = simpledialog.askinteger(title="Single box or files?", prompt="Manualy enter box (0) or read several boxes from csv? (1)"
                                            , initialvalue=0, maxvalue=1, minvalue=0, parent=gui.root)
        strip_by_box = bool(strip_by_box)
        if strip_by_box:
            # ask for filename, class number, output name
            box_file_path = filedialog.askopenfilename(defaultextension=".csv", title="Select the box file to open")
            new_file_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Enter name for created training data file")
            class_number = simpledialog.askinteger(title="Class number", prompt="Enter the number for the class of pixel in the boxes given", parent=gui.root)

            def worker():
                try:
                    gui.root.after(0, lambda: gui.write_to_terminal("Starting box stripping"))
                    brain.strip_from_boxes(box_file_path, class_number, new_file_path)
                    gui.root.after(0, lambda: gui.write_to_terminal(f"Stripping complete. File saved to {new_file_path}"))

                except Exception as e:
                    gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to strip from boxes: {e}"))
            thread1 = threading.Thread(target=worker, daemon=True)
        else: # not strip by box file
            output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Select output filename")

            lat_high = simpledialog.askfloat(title="lat high", prompt="Enter maximum latitude", parent=gui.root)
            lon_low = simpledialog.askfloat(title="lon low", prompt="Enter minimum longitude", parent=gui.root)
            lat_low = simpledialog.askfloat(title="lat low", prompt="Enter minimum latitude", parent=gui.root)
            lon_high = simpledialog.askfloat(title="lon high", prompt="Enter maximum longitude", parent=gui.root)

            class_number = simpledialog.askinteger(title="class number", prompt="Enter the number for the class of pixel in the box given", parent=gui.root)

            def worker():
                try:
                    gui.root.after(0, lambda: gui.write_to_terminal("Writing from box"))
                    brain.create_training_data_from_box(output_file_path, lat_high, lat_low, lon_high, lon_low, class_number)
                    gui.root.after(0, lambda: gui.write_to_terminal("Complete"))
                except Exception as e:
                    gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to create training data: {e}"))
            thread1 = threading.Thread(target=worker, daemon=True)

    else: # create from truth data
        if brain.truth_arr is None: # only ask for new input file if one is not loaded
            input_file_path = filedialog.askopenfilename(defaultextension=".csv", title="Select the truth data file to open")
        else:
            input_file_path = "" # use blank string since this is only accessed when no file is loaded
        output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Enter name for created training data file")
        gui.root.after(0, lambda: gui.write_to_terminal("Reading in truth data and creating csv file"))
        def worker():
            try:
                brain.write_truth_to_csv(input_file_path, output_file_path)
                gui.root.after(0, lambda: gui.write_to_terminal(f"Operation complete. Training data written to {output_file_path}"))

                if input_file_path != "":
                    truth_name = os.path.basename(input_file_path)
                    gui.root.after(0, lambda: gui.write_to_terminal("Complete"))
                    gui.root.after(0, lambda: gui.DOC_file_label.config(text=f"Truth file: {truth_name}"))
            except Exception as e:
                gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Failed to complete operation: {e}"))
        
        thread1 = threading.Thread(target=worker, daemon=True)
                

    thread1.start()
    check_thread_progress(thread1)


def make_predictions():
    gui.write_to_terminal("NOTE: tif files produced will use the currently loaded HDF file's transform")
    output_path = filedialog.asksaveasfilename(title="Enter name for output tif file")

    def worker():
        try:
            predictions = brain.make_predictions()

            gui.root.after(0, lambda:gui.write_to_terminal("Writing predictions"))
            brain.write_predictions(predictions, output_path)

            if predictions[1] is not None: # avoids running mean filter on regressions
                gui.root.after(0, lambda:gui.write_to_terminal("Generating post-processed predictions."))
                gui.root.after(0, lambda:gui.write_to_terminal("Writing mean filter version"))
                brain.write_mean_filter_pred(predictions[0], output_path) # choose predictions[0] to only get class predictions results

            gui.root.after(0, lambda:gui.write_to_terminal("Results saved to disk"))
        except Exception as e:
            gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to complete predictions or subtask. {e}"))

    gui.root.after(0, lambda: gui.write_to_terminal("Producing initial predictions"))
    thread1 = threading.Thread(target=worker, daemon=True)
    thread1.start()
    check_thread_progress(thread1)



def load_model():
    model_path = filedialog.askopenfilename(defaultextension=".joblib", title="Select the model file to open")
    try:
        brain.load_old_model(model_path)
        gui.root.after(0, lambda: gui.write_to_terminal(f"Loaded model at location: {model_path}"))
    except Exception as e:
        gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to load model: {e}"))


def save_model():
    try:
        model_path = filedialog.asksaveasfilename(title="Enter filename for model", defaultextension=".joblib")
        brain.write_model(model_path)
        gui.root.after(0, lambda: gui.write_to_terminal(f"Model written to disk as {model_path}"))
    except Exception as e:
        gui.root.after(0, lambda e=e: gui.write_to_terminal(f"Unable to write to disk: {e}"))

# configure GUI buttons to work with backend logic
gui.modis_button.config(command=read_in_modis)
gui.doc_button.config(command=read_in_truth)
gui.gen_data_button.config(command=create_training_data)
gui.train_model_button.config(command=train_model_csv)
gui.make_prediction_button.config(command=make_predictions)
gui.load_model_button.config(command=load_model)
gui.save_model_button.config(command=save_model)

# start GUI
gui.root.mainloop()
