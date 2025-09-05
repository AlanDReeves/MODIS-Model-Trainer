import tkinter as tk

class GenGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MODIS 500m Model Trainer")
        self.root.geometry("940x540")

        left_menu = tk.Frame(self.root)
        left_menu.pack(side="left", fill="y", pady=10, padx=20)
        bottom_menu = tk.Frame(left_menu)
        bottom_menu.pack(side="bottom", pady=10)
        top_menu = tk.Frame(left_menu)
        top_menu.pack(side="top", pady=10)

        right_frame = tk.Frame(self.root)
        right_frame.pack(side="right", fill="x")


        self.terminal = tk.Text(master=right_frame, bg="black", fg="lime", insertbackground="white", height=540)
        self.terminal.pack(fill="y", side="right")
        self.terminal.insert("end", "Model Trainer graphical interface.\nPlease choose an option to begin\n")
        self.terminal.config(state="disabled") # to prevent the user from writing

        #bottom menu
        self.modis_button = tk.Button(master=bottom_menu, text="Load MODIS file")
        self.modis_button.pack(fill="x", side="bottom", pady=1)
        self.doc_button = tk.Button(master=bottom_menu, text="Load truth data file")
        self.doc_button.pack(fill="x", side="bottom", pady=1)
        self.gen_data_button = tk.Button(master=bottom_menu, text="Generate training data")
        self.gen_data_button.pack(fill="x", side="bottom", pady=1)
        self.train_model_button = tk.Button(master=bottom_menu, text="Train a Model")
        self.train_model_button.pack(fill="x", side="bottom", pady=1)


        #top menu
        self.MODIS_file_label = tk.Label(master=top_menu, text="No MODIS File Loaded")
        self.MODIS_file_label.pack(fill="x", side="top")
        self.DOC_file_label = tk.Label(master=top_menu, text="No Truth File Loaded")
        self.DOC_file_label.pack(fill="x", side="top")
        self.make_prediction_button = tk.Button(master=top_menu, text="Make Prediction for current HDF")
        self.make_prediction_button.pack(fill='x', side="top", pady=1)
        self.load_model_button = tk.Button(master=top_menu, text="Load a saved model")
        self.load_model_button.pack(fill="x", side="bottom", pady=1)
        self.save_model_button = tk.Button(master=top_menu, text="Save current model to disk")
        self.save_model_button.pack(fill="x", side="bottom", pady=1)

    def write_to_terminal(self, text):
        self.terminal.config(state="normal")
        self.terminal.insert("end", text + '\n')
        self.terminal.see("end")
        self.terminal.config(state="disabled")

    def write_to_terminal_same_line(self, text):
        self.terminal.config(state="normal")
        self.terminal.insert("end", text)
        self.terminal.see("end")
        self.terminal.config(state="disabled")

if __name__ == "__main__":
    gui = GenGUI()
    gui.root.mainloop()
