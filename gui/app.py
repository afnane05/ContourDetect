import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from methodes.sobel import apply_sobel
from methodes.prewitt import apply_prewitt
from methodes.canny import apply_canny
from methodes.laplacien import apply_laplacian
from utils.gestion_fichiers import save_image, load_image
import os

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Analysis Tool")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.original_image = None
        self.filtered_image = None
        self.original_photo = None
        self.filtered_photo = None
        self.original_cv_image = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Create main frames
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        image_frame = ttk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Control panel
        self.create_control_panel(control_frame)
        
        # Image display frames
        self.create_image_displays(image_frame)
    
    def create_control_panel(self, parent):
        # Upload button
        upload_btn = ttk.Button(parent, text="Upload Image", 
                               command=self.upload_image, style="Accent.TButton")
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Filter selection
        ttk.Label(parent, text="Filter:").pack(side=tk.LEFT, padx=5)
        
        self.filter_var = tk.StringVar(value="Sobel")
        filter_combo = ttk.Combobox(parent, textvariable=self.filter_var,
                                   values=["Sobel", "Prewitt", "Canny", "Laplacien"],
                                   state="readonly", width=15)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind('<<ComboboxSelected>>', self.apply_filter)
        
        # Canny threshold controls (initially hidden)
        self.canny_frame = ttk.Frame(parent)
        
        ttk.Label(self.canny_frame, text="Threshold1:").pack(side=tk.LEFT)
        self.threshold1_var = tk.IntVar(value=100)
        threshold1_spin = ttk.Spinbox(self.canny_frame, from_=0, to=500, 
                                     textvariable=self.threshold1_var, width=5)
        threshold1_spin.pack(side=tk.LEFT, padx=2)
        threshold1_spin.bind('<Return>', self.apply_filter)
        
        ttk.Label(self.canny_frame, text="Threshold2:").pack(side=tk.LEFT)
        self.threshold2_var = tk.IntVar(value=200)
        threshold2_spin = ttk.Spinbox(self.canny_frame, from_=0, to=500, 
                                     textvariable=self.threshold2_var, width=5)
        threshold2_spin.pack(side=tk.LEFT, padx=2)
        threshold2_spin.bind('<Return>', self.apply_filter)
        
        # Save button
        save_btn = ttk.Button(parent, text="Save Result", 
                             command=self.save_result, style="Accent.TButton")
        save_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_image_displays(self, parent):
        # Left frame for original image
        self.original_frame = ttk.LabelFrame(parent, text="Original Image", padding="10")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.original_label = ttk.Label(self.original_frame, text="No image loaded", 
                                       style="ImageLabel.TLabel")
        self.original_label.pack(expand=True)
        
        # Right frame for filtered image
        self.filtered_frame = ttk.LabelFrame(parent, text="Filtered Image", padding="10")
        self.filtered_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.filtered_label = ttk.Label(self.filtered_frame, text="Apply a filter to see result", 
                                       style="ImageLabel.TLabel")
        self.filtered_label.pack(expand=True)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_cv_image = load_image(file_path)
                self.original_image = Image.fromarray(cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB))
                self.display_images()
                self.apply_filter()
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_images(self):
        # Display original image
        if self.original_image:
            # Resize for display while maintaining aspect ratio
            display_size = (400, 400)
            original_display = self.original_image.copy()
            original_display.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            self.original_photo = ImageTk.PhotoImage(original_display)
            self.original_label.configure(image=self.original_photo, text="")
        
        # Display filtered image
        if self.filtered_image:
            filtered_display = self.filtered_image.copy()
            filtered_display.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            self.filtered_photo = ImageTk.PhotoImage(filtered_display)
            self.filtered_label.configure(image=self.filtered_photo, text="")
    
    def apply_filter(self, event=None):
        if self.original_cv_image is None:
            return

        selected_filter = self.filter_var.get()

        try:
            # Apply selected filter
            if selected_filter == "Sobel":
                filtered_array = apply_sobel(self.original_cv_image)
            elif selected_filter == "Prewitt":
                filtered_array = apply_prewitt(self.original_cv_image)
            elif selected_filter == "Canny":
                # Call Canny WITHOUT thresholds → uses OpenCV-like auto thresholds
                filtered_array = apply_canny(self.original_cv_image)
            elif selected_filter == "Laplacien":
                filtered_array = apply_laplacian(self.original_cv_image)
            else:
                return

            # Convert to PIL Image for display
            self.filtered_image = Image.fromarray(filtered_array)
            self.display_images()

        except Exception as e:
            tk.messagebox.showerror("Error", f"Filter application failed: {str(e)}")

    def save_result(self):
        if self.filtered_image is None:
            tk.messagebox.showwarning("Warning", "No filtered image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Filtered Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                save_image(self.filtered_image, file_path)
                tk.messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to save image: {str(e)}")