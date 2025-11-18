import tkinter as tk
from tkinter import ttk

def configure_styles():
    style = ttk.Style()
    
    # Configure main styles
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
    style.configure("TButton", font=("Arial", 10), padding=6)
    style.configure("Accent.TButton", background="#007acc", foreground="white")
    style.configure("TCombobox", font=("Arial", 10))
    style.configure("TLabelframe", background="#f0f0f0", font=("Arial", 11, "bold"))
    style.configure("TLabelframe.Label", background="#f0f0f0", font=("Arial", 11, "bold"))
    
    # Custom style for image labels
    style.configure("ImageLabel.TLabel", 
                   background="#ffffff",
                   relief="solid",
                   borderwidth=1,
                   anchor="center",
                   font=("Arial", 10, "italic"))
    
    # Configure spinbox
    style.configure("TSpinbox", font=("Arial", 9))
    
    return style