import numpy as np
from PIL import Image, ImageGrab, ImageOps
import tkinter as tk

# defining the class for the user drawing    
class DigitalDrawer:
    # initialize the DigitalDrawer object
    def __init__(self, width, height, brush_color, model):
        self.w = width
        self.h = height
        self.color = brush_color
        self.model = model

        # initialize the main window with buttons
        self.root = tk.Tk()
        self.root.title("Sketch a Digit")
        self.predict_button = tk.Button(master=self.root, text="Predict", command=self.finalize_drawing)
        self.clear_button = tk.Button(master=self.root, text="Clear", command=self.clear_canvas)
        self.prediction_label = tk.Label(master=self.root, text="", font=("Helvetica", 16))

        # create the actual canvas popup
        self.canvas = tk.Canvas(self.root, width = self.w, height = self.h, bg = 'black')
        self.canvas.pack()

        # initialize the image matrix to zeros
        self.image_matrix = np.zeros((28,28), dtype = float)

    # turns the finalized canvas into an image
    def grab_and_process_image(self):
        # gets canvas location
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # grab image from screen for the canvas area
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    
        # convert to grayscale
        img = img.convert('L')

        # resize the image
        resized_img = img.resize((28, 28), Image.Resampling.LANCZOS)

        return resized_img
    
    # turns the user's drawing into a 28x28 matrix representation
    def finalize_drawing(self):
        img = self.grab_and_process_image()
        self.image_matrix = np.array(img) / 255.0
        self.predict_digit()
    
    # clears the canvas and image matrix
    def clear_canvas(self):
        self.canvas.delete("all") 
        self.image_matrix = np.zeros((28, 28), dtype=float)
        self.prediction_label.config(text=f"Predicted Digit: ")

    # implements the paintbrush function for the canvas
    def draw(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.create_oval(self.x - 4, self.y - 4, self.x + 4, self.y + 4, fill=self.color, outline=self.color)
    
    # runs the actual prediction
    def predict_digit(self):
        image_flat = self.image_matrix.flatten().reshape(1, -1)
        output = self.model.forward(image_flat)
        prediction = np.argmax(output)
        self.prediction_label.config(text=f"Predicted Digit: {prediction}")

    # runner function
    def run(self):
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button.pack(side="right", padx=10)
        self.clear_button.pack(side="right", padx=10)
        self.prediction_label.pack(side='bottom', padx=10)

        self.root.mainloop()