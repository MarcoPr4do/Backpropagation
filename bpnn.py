from tkinter import *
import numpy
import pandas


breast_cancer_df =  pandas.read_csv("data.csv", usecols=[i for i in range(0, 32)])
inputs_array = breast_cancer_df.drop("diagnosis",axis=1).values
outputs_array = breast_cancer_df["diagnosis"].values
outputs_array[outputs_array == 'B'] = 0
outputs_array[outputs_array == 'M'] = 1

input_training = inputs_array[:512]
input_test = inputs_array[512:]
output_training = outputs_array[:512]
output_test = outputs_array[512:]
input_max = input_training.max(axis=0)
input_min = input_training.min(axis=0)

input_training = (input_training - input_min) / (input_max - input_min)
input_test = (input_test - input_min) / (input_max - input_min)
class Network:
    def __init__(self, neurons):
        self.hidden_layer_neurons=neurons
    def randomize_weights(self, input_size, hidden_neurons):
        self.hidden_weights = numpy.random.randn(input_size, hidden_neurons)
        self.hidden_bias=numpy.zeros(hidden_neurons)
        self.output_weights=numpy.random.randn(hidden_neurons,1)
        self.output_bias=numpy.random.randn(1)
    def calc_error(self,predicted_output,expected_output):
        errSum = 0.0
        for i in range(len(expected_output)):
            err = predicted_output[i] - expected_output[i]
            errSum += 0.5 * err * err
        return errSum
    def accuracy(self, expected, prediction):
        return numpy.sum(expected==prediction)/len(expected)
    def relu(self, input_x):
        return numpy.maximum(0, input_x)
    def relu_derivative(self, Z):
        dZ = numpy.zeros(Z.shape)
        dZ[Z>0] = 1
        return dZ
    def sigmoid(self, y):
        return 1/(1+numpy.power(numpy.e,-y))
    def forward_propagation(self, inputs):
        hidden_layer_activation = numpy.dot(inputs,self.hidden_weights)+self.hidden_bias
        hidden_layer_output = self.relu(hidden_layer_activation)
        output_layer_activation = numpy.dot(hidden_layer_output,self.output_weights)+self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)
        self.forward_cache = (hidden_layer_activation, hidden_layer_output, output_layer_activation, predicted_output)
        return predicted_output.ravel()
    def prediction(self, inputs, flag):
        probability = self.forward_propagation(inputs)
        predicted_output = numpy.zeros(inputs.shape[0], dtype=int)
        predicted_output[probability>=0.5]=1
        predicted_output[probability<0.5]=0
        if(flag):
            return (predicted_output, probability)
        return predicted_output
    def training(self, inputs, outputs, epochs, learning_rate):
        self.randomize_weights(len(inputs[0]), self.hidden_layer_neurons)
        for i in range(epochs):
            Y = self.forward_propagation(inputs)
            d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias = self.backpropagation(inputs, outputs)
            self.hidden_weights = self.hidden_weights-learning_rate*d_hidden_weights
            self.hidden_bias = self.hidden_bias-learning_rate*d_hidden_bias
            self.output_weights = self.output_weights-learning_rate*d_output_weights
            self.output_bias = self.output_bias-learning_rate*d_output_bias
            if(i%(epochs/10)==0):
                predicted_output=self.prediction(inputs,False)
                print("Epoch {} MSE: {}".format(i,self.calc_error(predicted_output,outputs)/len(outputs)))
        predicted_output=self.prediction(inputs,False)
        print("Trained with MSE: {}".format(self.calc_error(predicted_output,outputs)/len(outputs)))
    def backpropagation(self, inputs, outputs):
        hla, hlo, ola, po = self.forward_cache
        hlo_size = len(hlo)
        error = po-outputs.reshape(-1,1)
        d_output_weights = numpy.dot(hlo.T, error)/hlo_size
        d_output_bias = numpy.sum(d_output_weights, axis=0)/hlo_size
        d_hidden_layer = numpy.dot(error, self.output_weights.T)*self.relu_derivative(hla)
        d_hidden_weights = numpy.dot(inputs.T, d_hidden_layer)/hlo_size
        d_hidden_bias = numpy.sum(d_hidden_layer, axis=0)/hlo_size
        return d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias
    def clasification(self, inputs, outputs):
        prediction, probability = self.prediction(inputs, True)
        accuracy = self.accuracy(outputs, prediction)
        return (accuracy, prediction)
def main():
    nn=Network(10)
    nn.training(input_training,output_training,1000,0.04)
    print("Esperado\tPredicción")
    accuracy,prediction=nn.clasification(input_test,output_test)
    for i in range(len(output_test)):
        print("{}\t\t{}".format(output_test[i],prediction[i]))
    print("Precisión: {}".format(accuracy))
# main()


window = Tk()
window.geometry("500x500")
window.title("Welcome to LikeGeeks app")
window.mainloop()