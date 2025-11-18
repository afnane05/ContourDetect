import tkinter as tk
from gui.app import ImageFilterApp
from gui.style import configure_styles

def main():
    root = tk.Tk()
    configure_styles()
    app = ImageFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()