import tkinter as tk
from tkinter import ttk

def select_bands(root, num_bands=7):
    popup = tk.Toplevel(root)
    popup.title("Select Bands")

    band_vars = []

    for i in range(num_bands):
        var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(popup, text=f"Band {i+1}", variable=var)
        chk.pack(anchor="w", padx=10, pady=2)
        band_vars.append((i, var))

    selected = []

    def submit():
        nonlocal selected
        selected = [i for i, var in band_vars if var.get()]
        popup.destroy()

    submit_btn = ttk.Button(popup, text="OK", command=submit)
    submit_btn.pack(pady=10)

    # make this popup modal (block until closed)
    popup.grab_set()
    root.wait_window(popup)

    return selected