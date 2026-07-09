# Neural Network from Scratch

This project implements a fully connected neural network in pure Python with the use of NumPy. Used to develop fluency in fundamental deep learning concepts: backpropogation, activation functions, initialization, batching techniques. The network is trained and evaluated on the MNIST dataset.

### Model Architecture

Our network is a two layer fully connected neural network using the sigmoid activation in its hidden layer, and softmax activation to generate the final layer probabilities. A brief outline:

vectorized inputs -> first linear layer -> sigmoid activation -> second linear layer -> softmax output -> argmax prediction

We train with vanilla gradient descent, all derivatives are hand implemented and gradients computed manually. Code is all included in nerual_net.py.

### Key Techniques
- Weights are initialized with Xavier initilaization to combat exploding/vanishing gradient problem (better suited for sigmoid networks)
- We perform mini-batch gradient descent with shuffled training data
- Eval on MNIST test set

### Training

Model is trained for 100 epochs on the MNIST train set with cross-entropy loss, which was enough for training to converge. We achieve just under 97% accuracy on the MNIST test set (multiple training runs converging to 96.7% accuracy on the test set).

### Discussion

This was my first ever proper CS project and my first time coding in Python in about three years. It was more than interesting to get a hands-on approach on neural networks and it helped hone in the key concepts that make up a simple neural network. Some key observations/learning moments I made along the way:

1. Adjusting the learning rate really made a big difference. I started out with 0.01 and the model plateued pretty quickly and after some tinkering, 0.05 got me the best accuracy.

2. There was a big issue with my original code, since I applied the sigmoid function on my output layer, which essentially broke all of the math. I spent a long time wondering why my code was so far off, until a good friend of mine pointed out that I was suppose to use softmax since only in that case would the derivatives nicely cancel out into y_pred - y_true. This connects to my third point...

3. Doing the math by hand to really understand the partial derivatives was key to getting at the heart of what makes this model learn. Also, I love math (more than coding at least) so it was cool for this project to be somewhat math invovled.

Concepts to study in the future:
- Data agumentation and other data curation techniques
- More complex architectures (CNN, RNN, Transformer)
- Other optimization algorithms
- Learn about other activation functions, their advantages and use cases

# Reproducibility

1. Create a virtual environment
    python3 -m venv venv
2. Activate
    source venv/bin/activate
3. Install dependecies
    pip install -r requirements.txt
4. Run training loop
    python main.py

### GUI Instructions

Credit to ChatGPT for generating the GUI; all other programs are hand implemented.

- Draw by dragging left click
- Click "Predict" to get model prediction
- Click "Clear" to clear canvas and prediction

### Credits/Acknowledgments
SHOUTOUT TO:
- MNIST dataset from Yann LeCun et al.
- Tkinter for GUI development
- NumPy for numerical operations

### LICENSE
This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.