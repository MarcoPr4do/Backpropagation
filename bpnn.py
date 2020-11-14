from tkinter import *
import numpy
import pandas

breast_cancer_df = pandas.read_csv(
    "data.csv", usecols=[i for i in range(0, 32)])
inputs_array = breast_cancer_df.drop("diagnosis", axis=1).values
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
        self.hidden_layer_neurons = neurons

    def randomize_weights(self, input_size, hidden_neurons):
        self.hidden_weights = numpy.random.randn(input_size, hidden_neurons)
        self.hidden_bias = numpy.zeros(hidden_neurons)
        self.output_weights = numpy.random.randn(hidden_neurons, 1)
        self.output_bias = numpy.random.randn(1)

    def calc_error(self, predicted_output, expected_output):
        errSum = 0.0
        for i in range(len(expected_output)):
            err = predicted_output[i] - expected_output[i]
            errSum += 0.5 * err * err
        return errSum

    def accuracy(self, expected, prediction):
        return numpy.sum(expected == prediction) / len(expected)

    def relu(self, input_x):
        return numpy.maximum(0, input_x)

    def relu_derivative(self, Z):
        dZ = numpy.zeros(Z.shape)
        dZ[Z > 0] = 1
        return dZ

    def sigmoid(self, y):
        return 1 / (1 + numpy.power(numpy.e, -y))

    def forward_propagation(self, inputs):
        hidden_layer_activation = numpy.dot(
            inputs, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = self.relu(hidden_layer_activation)
        output_layer_activation = numpy.dot(
            hidden_layer_output, self.output_weights) + self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)
        self.forward_cache = (hidden_layer_activation, hidden_layer_output,
                              output_layer_activation, predicted_output)
        return predicted_output.ravel()

    def prediction(self, inputs, flag):
        probability = self.forward_propagation(inputs)
        predicted_output = numpy.zeros(inputs.shape[0], dtype=int)
        predicted_output[probability >= 0.5] = 1
        predicted_output[probability < 0.5] = 0
        if (flag):
            return (predicted_output, probability)
        return predicted_output

    def training(self, inputs, outputs, epochs, learning_rate):
        self.randomize_weights(len(inputs[0]), self.hidden_layer_neurons)
        for i in range(epochs):
            Y = self.forward_propagation(inputs)
            d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias = self.backpropagation(
                inputs, outputs)
            self.hidden_weights = self.hidden_weights - learning_rate * d_hidden_weights
            self.hidden_bias = self.hidden_bias - learning_rate * d_hidden_bias
            self.output_weights = self.output_weights - learning_rate * d_output_weights
            self.output_bias = self.output_bias - learning_rate * d_output_bias
            if (i % (epochs / 10) == 0):
                predicted_output = self.prediction(inputs, False)
                print("Epoch {} MSE: {}".format(i, self.calc_error(
                    predicted_output, outputs) / len(outputs)))
        predicted_output = self.prediction(inputs, False)
        print("Trained with MSE: {}".format(
            self.calc_error(predicted_output, outputs) / len(outputs)))

    def backpropagation(self, inputs, outputs):
        hla, hlo, ola, po = self.forward_cache
        hlo_size = len(hlo)
        error = po - outputs.reshape(-1, 1)
        d_output_weights = numpy.dot(hlo.T, error) / hlo_size
        d_output_bias = numpy.sum(d_output_weights, axis=0) / hlo_size
        d_hidden_layer = numpy.dot(
            error, self.output_weights.T) * self.relu_derivative(hla)
        d_hidden_weights = numpy.dot(inputs.T, d_hidden_layer) / hlo_size
        d_hidden_bias = numpy.sum(d_hidden_layer, axis=0) / hlo_size
        return d_hidden_weights, d_hidden_bias, d_output_weights, d_output_bias

    def clasification(self, inputs, outputs):
        prediction, probability = self.prediction(inputs, True)
        accuracy = self.accuracy(outputs, prediction)
        return (accuracy, prediction)


def main():
    nn = Network(10)
    nn.training(input_training, output_training, 1000, 0.04)
    print("Esperado\tPredicción")
    accuracy, prediction = nn.clasification(input_test, output_test)
    for i in range(len(output_test)):
        print("{}\t\t{}".format(output_test[i], prediction[i]))
    print("Precisión: {}".format(accuracy))


# main()


# GUI
window = Tk()
window.geometry("600x700")
window.title("Welcome to LikeGeeks app")
heading = Label(text="Backpropagation", bg="grey",
                fg="black", width="500", height="3")
heading.pack()

media_del_radio_text = Label(text="media_del_radio *", )
textura_media_text = Label(text="textura_media *", )
media_del_perímetro_text = Label(text="media_del_perímetro *", )
media_del_área_text = Label(text="media_del_área *", )
suavidad_significa_text = Label(text="suavidad_significa *", )
compacidad_significa_text = Label(text="compacidad_significa *", )
media_de_concavidad_text = Label(text="media_de_concavidad *", )
puntos_cóncavos_significan_text = Label(text="puntos_cóncavos_significan *", )
simetría_media_text = Label(text="simetría_media *", )
media_de_la_dimensión_fractal_text = Label(
    text="media_de_la_dimensión_fractal *", )
radio_se_text = Label(text="radio_se *", )
textura_se_text = Label(text="textura_se *", )
perímetro_se_text = Label(text="perímetro_se *", )
area_se_text = Label(text="area_se *", )
suavidad_se_text = Label(text="suavidad_se *", )
compacidad_se_text = Label(text="compacidad_se *", )
concavidad_se_text = Label(text="concavidad_se *", )
puntos_concavos_se_text = Label(text="puntos_concavos_se *", )
simetría_se_text = Label(text="simetría_se *", )
dimension_fractal_se_text = Label(text="dimension_fractal_se *", )
radio_peor_text = Label(text="radio_peor *", )
textura_peor_text = Label(text="textura_peor *", )
perímetro_peor_text = Label(text="perímetro_peor *", )
zona_peor_text = Label(text="zona_peor *", )
suavidad_peor_text = Label(text="suavidad_peor *", )
compacidad_peor_text = Label(text="compacidad_peor *", )
peor_concavidad_text = Label(text="peor_concavidad *", )
puntos_concavos_peor_text = Label(text="puntos_concavos_peor *", )
simetria_peor_text = Label(text="simetria_peor *", )
dimension_fractal_peor_text = Label(text="dimension_fractal_peor *", )

media_del_radio_text.place(x=15, y=70)
textura_media_text.place(x=15, y=90)
media_del_perímetro_text.place(x=15, y=110)
media_del_área_text.place(x=15, y=130)
suavidad_significa_text.place(x=15, y=150)
compacidad_significa_text.place(x=15, y=170)
media_de_concavidad_text.place(x=15, y=190)
puntos_cóncavos_significan_text.place(x=15, y=210)
simetría_media_text.place(x=15, y=230)
media_de_la_dimensión_fractal_text.place(x=15, y=250)
radio_se_text.place(x=15, y=270)
textura_se_text.place(x=15, y=290)
perímetro_se_text.place(x=15, y=310)
area_se_text.place(x=15, y=330)
suavidad_se_text.place(x=15, y=350)
compacidad_se_text.place(x=15, y=370)
concavidad_se_text.place(x=15, y=390)
puntos_concavos_se_text.place(x=15, y=410)
simetría_se_text.place(x=15, y=430)
dimension_fractal_se_text.place(x=15, y=450)
radio_peor_text.place(x=15, y=470)
textura_peor_text.place(x=15, y=490)
perímetro_peor_text.place(x=15, y=510)
zona_peor_text.place(x=15, y=530)
suavidad_peor_text.place(x=15, y=550)
compacidad_peor_text.place(x=15, y=570)
peor_concavidad_text.place(x=15, y=590)
puntos_concavos_peor_text.place(x=15, y=610)
simetria_peor_text.place(x=15, y=630)
dimension_fractal_peor_text.place(x=15, y=650)

media_del_radio = DoubleVar()
textura_media = DoubleVar()
media_del_perímetro = DoubleVar()
media_del_área = DoubleVar()
suavidad_significa = DoubleVar()
compacidad_significa = DoubleVar()
media_de_concavidad = DoubleVar()
puntos_cóncavos_significan = DoubleVar()
simetría_media = DoubleVar()
media_de_la_dimensión_fractal = DoubleVar()
radio_se = DoubleVar()
textura_se = DoubleVar()
perímetro_se = DoubleVar()
area_se = DoubleVar()
suavidad_se = DoubleVar()
compacidad_se = DoubleVar()
concavidad_se = DoubleVar()
puntos_concavos_se = DoubleVar()
simetría_se = DoubleVar()
dimension_fractal_se = DoubleVar()
radio_peor = DoubleVar()
textura_peor = DoubleVar()
perímetro_peor = DoubleVar()
zona_peor = DoubleVar()
suavidad_peor = DoubleVar()
compacidad_peor = DoubleVar()
peor_concavidad = DoubleVar()
puntos_concavos_peor = DoubleVar()
simetria_peor = DoubleVar()
dimension_fractal_peor = DoubleVar()

media_del_radio_entry = Entry(textvariable=media_del_radio)
textura_media_entry = Entry(textvariable=textura_media)
media_del_perímetro_entry = Entry(textvariable=media_del_perímetro)
media_del_área_entry = Entry(textvariable=media_del_área)
suavidad_significa_entry = Entry(textvariable=suavidad_significa)
compacidad_significa_entry = Entry(textvariable=compacidad_significa)
media_de_concavidad_entry = Entry(textvariable=media_de_concavidad)
puntos_cóncavos_significan_entry = Entry(
    textvariable=puntos_cóncavos_significan)
simetría_media_entry = Entry(textvariable=simetría_media)
media_de_la_dimensión_fractal_entry = Entry(
    textvariable=media_de_la_dimensión_fractal)
radio_se_entry = Entry(textvariable=radio_se)
textura_se_entry = Entry(textvariable=textura_se)
perímetro_se_entry = Entry(textvariable=perímetro_se)
area_se_entry = Entry(textvariable=area_se)
suavidad_se_entry = Entry(textvariable=suavidad_se)
compacidad_se_entry = Entry(textvariable=compacidad_se)
concavidad_se_entry = Entry(textvariable=concavidad_se)
puntos_concavos_se_entry = Entry(textvariable=puntos_concavos_se)
simetría_se_entry = Entry(textvariable=simetría_se)
dimension_fractal_se_entry = Entry(textvariable=dimension_fractal_se)
radio_peor_entry = Entry(textvariable=radio_peor)
textura_peor_entry = Entry(textvariable=textura_peor)
perímetro_peor_entry = Entry(textvariable=perímetro_peor)
zona_peor_entry = Entry(textvariable=zona_peor)
suavidad_peor_entry = Entry(textvariable=suavidad_peor)
compacidad_peor_entry = Entry(textvariable=compacidad_peor)
peor_concavidad_entry = Entry(textvariable=peor_concavidad)
puntos_concavos_peor_entry = Entry(textvariable=puntos_concavos_peor)
simetria_peor_entry = Entry(textvariable=simetria_peor)
dimension_fractal_peor_entry = Entry(textvariable=dimension_fractal_peor)

media_del_radio_entry.place(x=195, y=70)
textura_media_entry.place(x=195, y=90)
media_del_perímetro_entry.place(x=195, y=110)
media_del_área_entry.place(x=195, y=130)
suavidad_significa_entry.place(x=195, y=150)
compacidad_significa_entry.place(x=195, y=170)
media_de_concavidad_entry.place(x=195, y=190)
puntos_cóncavos_significan_entry.place(x=195, y=210)
simetría_media_entry.place(x=195, y=230)
media_de_la_dimensión_fractal_entry.place(x=195, y=250)
radio_se_entry.place(x=195, y=270)
textura_se_entry.place(x=195, y=290)
perímetro_se_entry.place(x=195, y=310)
area_se_entry.place(x=195, y=330)
suavidad_se_entry.place(x=195, y=350)
compacidad_se_entry.place(x=195, y=370)
concavidad_se_entry.place(x=195, y=390)
puntos_concavos_se_entry.place(x=195, y=410)
simetría_se_entry.place(x=195, y=430)
dimension_fractal_se_entry.place(x=195, y=450)
radio_peor_entry.place(x=195, y=470)
textura_peor_entry.place(x=195, y=490)
perímetro_peor_entry.place(x=195, y=510)
zona_peor_entry.place(x=195, y=530)
suavidad_peor_entry.place(x=195, y=550)
compacidad_peor_entry.place(x=195, y=570)
peor_concavidad_entry.place(x=195, y=590)
puntos_concavos_peor_entry.place(x=195, y=610)
simetria_peor_entry.place(x=195, y=630)
dimension_fractal_peor_entry.place(x=195, y=650)

# Salida de respuesta
output_text = Label(text="", )
output_text.place(x=370, y=130)


def save_info():
    print("press btn")
    # cambia el text del resultado
    output_text.config(text="boton presionado")
    print(["123  1234  1236"])
    print(input_test)


testerBtn = Button(window, text="Testear", width="20", height="2", command=save_info)
testerBtn.place(x=370, y=78)

window.mainloop()
