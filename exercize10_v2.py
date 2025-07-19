import numpy as np

"""
Using only numpy we will implement a neural network
"""


class NeuralNetwork():

    def __init__(self,
                 n_node_input: int,
                 n_nodes_hidden: list[int],
                 n_node_output: int,
                 activation: 'str' or callable = 'sigmoid',
                 bias=True
                 ):
        self.n_input = n_node_input
        self.n_hiddens = n_nodes_hidden
        self.n_output = n_node_output
        self.activation_function = activation
        self.biased = bias

    def sigmoid(self, x):
        sigma = 1 / (1 + np.exp(-x))
        return sigma

    def init_neural_network_weights(
            self,
            input_layer: int = None,
            hidden_layers: list[int] = None,
            output_layer: int = None,
    ):
        if input_layer is None:
            input_layer = self.n_input
        if hidden_layers is None:
            hidden_layers = self.n_hiddens
        if output_layer is None:
            output_layer = self.n_output


        bs = []
        # create layers:
        # create first layer:
        a1 = hidden_layers[0]
        a2 = input_layer
        w_in = np.full((a1, a2), 0.5)
        b_1 = np.full((a1, ), 1.0)
        bs.append(b_1)
        # create last layer
        an1 = output_layer
        an2 = hidden_layers[-1]
        w_out = np.full((an1, an2), 0.5)

        # create the hidden layers
        ws = []

        for i in range(len(hidden_layers) - 1):
            a1n = hidden_layers[i]
            a2n = hidden_layers[i+1]
            w_hidden_n = np.full((a2n, a1n), 0.5)
            b = np.full((a2n, ), 1.0)
            bs.append(b)
            ws.append(w_hidden_n)
        w_hidden = ws
        self.w_in = w_in
        self.w_out = w_out
        self.w_hidden = w_hidden
        self.b = bs
        return w_in, w_hidden, w_out, bs

    def forward_pass_single(self,
                            x: np.ndarray,
                            w_in: np.ndarray = None,
                            w_hidden: np.ndarray = None,
                            w_out: np.ndarray = None,
                            b: np.ndarray = None,
                            bias=None):

        if w_in is None:
            w_in = self.w_in
        if w_hidden is None:
            w_hidden = self.w_hidden
        if w_out is None:
            w_out = self.w_out
        if b is None:
            b = self.b
        if self.activation_function == 'sigmoid':
            activation = self.sigmoid
        else:
            activation = self.activation_function

        if bias is None:
            bias = self.biased

        if bias:
            bias = 1
        else:
            bias = 0
        z_1 = w_in @ x + bias * b[0]
        x_1 = activation(z_1)
        zs = []
        xs = []
        x_11 = x_1
        for i in range(len(w_hidden)):
            if i < len(w_hidden):
                z = w_hidden[i] @ x_11 + bias * b[i+1] #possible mistake in shape missmatch
            else:
                z = w_hidden[i] @ x_11
            zs.append(z)
            x = activation(z)
            xs.append(x)
            x_11 = x

        results = [x_1, z_1]
        for i in range(len(xs)):
            results.append(xs[i])
            results.append(zs[i])

        z_n = w_out @ x_11
        x_n = activation(z_n)

        results.append(x_n)
        results.append(z_n)
        self.last_result = results
        return results

    def forward_pass_batch(self,
                           x: np.ndarray,
                           w_in: np.ndarray = None,
                           w_hidden: np.ndarray = None,
                           w_out: np.ndarray = None,
                           b: np.ndarray = None,
                           bias=None
                           ):
        if w_in is None:
            w_in = self.w_in
        if w_hidden is None:
            w_hidden = self.w_hidden
        if w_out is None:
            w_out = self.w_out
        if b is None:
            b = self.b
        if self.activation_function == 'sigmoid':
            activation = self.sigmoid
        else:
            activation = self.activation_function

        if bias is None:
            bias = self.biased

        if bias:
            bias = 1
        else:
            bias = 0

        # do one single pass for each x_i in x

        batch_results = []
        for xi in x:
            result = self.forward_pass_single(xi, w_in, w_hidden, w_out, b, bias)

            batch_results.append(result)
        self.batch_results = batch_results
        return batch_results

    def get_batch_results(self):
        batch_results = self.batch_results
        outputs = []
        for results in batch_results:
            output = results[-2].item()
            outputs.append(output)
        return outputs

    def get_result(self):
        result = self.last_result
        output = result[-2]
        return output

    def binary_cross_entropy_loss(self,
                                  y_pred: float or list,
                                  y_true: float or list,
                                  eps = 1e-5):
        if y_pred is float:
            y_pred = [y_pred]
        if y_true is float:
            y_true = [y_true]
        elif y_true is np.ndarray:
            y_true = y_true.tolist(y_true)
        N = len(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.clip(y_pred, eps, 1-eps)
        loss = (-1 / N) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        #loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        # bunu nasil fark edicem ben bu ikisi ayni sey degilmis...
        return loss





if __name__ == '__main__':
    NN = NeuralNetwork(2, [3, 2], 1, activation='sigmoid')
    NN.init_neural_network_weights()
    results_batch_ = []
    array = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    x = array[:, :2]
    y = array[:, -1]

    for i, xi in enumerate(x):
        array = xi
        results = NN.forward_pass_single(array)
        print(results)
        result = NN.get_result()
        loss = NN.binary_cross_entropy_loss(result, np.array([y[i]]))
        results_batch_.append(loss)
    sigle_results = np.mean(results_batch_)
    print(f'single loss: {sigle_results}')


    NN.forward_pass_batch(x)
    results_batch = NN.get_batch_results()
    loss = NN.binary_cross_entropy_loss(results_batch[:], y)
    print(f'batch loss: {loss}')

# 0.7909960185139984












