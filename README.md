# Digit Classifier - Neural Network Built from Scratch 

This project is a completely connected neural network coded from scratch (no deep learning libraries like PyTorch or TensorFlow) which can recognize handwritten digits. The model is trained on the MNIST dataset and includes a Tkinter-based drawing canvas in which the user can test the model on their own hand drawn digits.

# Features

- Custom built neural network using NumPy
- Forward pass and backpropogation manually built
- Mini-batch gradient descent with shuffled training data
- Xavier initialization for more optimized training
- Tkinter GUI which allows user to test the model on their own handwritten digits
- Live feedback of model's loss and accuracy when testing on test set
- Achieves **over 97% accuracy** on MNIST test set

# Installation

1. Create a virtual environment:
    python3 -m venv venv
2. Activate it:
    source venv/bin/activate
3. Install dependecies
    pip install -r requirements.txt
4. Run the code
    python main.py

# GUI Instructions

- Draw by dragging left click
- Click "Predict" to get model prediction
- Click "Clear" to clear canvas and prediction

# Credits/Acknowledgments
SHOUTOUT TO:
- MNIST dataset from Yann LeCun et al.
- Tkinter for GUI development
- NumPy for numerical operations

# LICENSE
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

# Author's Comments and Future Work

This was my first ever proper CS project and my first time coding in Python in about three years. It was more than interesting to get a hands-on approach on neural networks and it helped hone in the key concepts that make up a simple neural network. Some key observations/learning moments I made along the way:

1. Adjusting the learning rate really made a big difference. I started out with 0.01 and the model plateued pretty quickly and after some tinkering, 0.05 got me the best accuracy.

2. There was a big issue with my original code, since I applied the sigmoid function on my output layer, which essentially broke all of the math. I spent a long time wondering why my code was so far off, until a good friend of mine pointed out that I was suppose to use softmax since only in that case would the derivatives nicely cancel out into y_pred - y_true. This connects to my third point...

3. Doing the math by hand to really understand the partial derivatives was key to getting at the heart of what makes this model learn. Also, I love math (more than coding at least) so it was cool for this project to be somewhat math invovled.
