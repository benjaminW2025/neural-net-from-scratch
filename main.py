from neural_net import NeuralNetwork
from canvas_gui import DigitalDrawer

def main():
    print("Start training...")

    # initialize the network
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

    # load data
    x_train = nn.load_images('train-images.idx3-ubyte')
    y_train = nn.load_labels('train-labels.idx1-ubyte')

    x_test = nn.load_images('t10k-images.idx3-ubyte')
    y_test = nn.load_labels('t10k-labels.idx1-ubyte')

    # train and test the model
    nn.train(x_train, y_train, x_test, y_test, epochs=100, batch_size=32, learning_rate=0.05)

    # allows user to draw digits on the canvas and model predicts
    drawing_canvas = DigitalDrawer(280, 280, 'white', nn)
    drawing_canvas.run()

if __name__ == "__main__":
    main()
