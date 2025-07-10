import numpy as np

array = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

x = array[:, :2]
y = array[:, -1]
input = np.concatenate((x, [[1], [1], [1], [1]]), 1)


###### Task 1 #######

def sigmoid_function(array: np.ndarray):
    sigmoid_temp = 1 + np.exp(- array)
    sigmoid = 1 / sigmoid_temp
    return sigmoid


###### Task 2 ######

"""
I will implement manual forward pass
Hidden layers: 2
Hidden layer 1: 3 neurons + bias, sigmoid
Hidden layer 2: 2 neurons + bias, sigmoid
Output layer: 1 neuron, sigmoid activation
"""

# weights_matrix_0_1 = np.ones((3, 3), dtype=float)
# weights_matrix_1_2 = np.ones((2, 4), dtype=float)
# weights_matrix_2_3 = np.ones((1, 2), dtype=float)

weights_matrix_0_1 = np.ones((3, 2), dtype=float)
weights_matrix_1_2 = np.ones((2, 3), dtype=float)
weights_matrix_2_3 = np.ones((1, 2), dtype=float)
bias_1 = np.array([1, 1, 1])
bias_2 = np.array([1, 1])

"""
performing forward pass now:
"""


def forward_pass(inputs: np.ndarray, weights: list[np.ndarray],
                 biases: list[np.ndarray]):
    """
    this is not right because the biases should always be 1 but this might get affected
    :param biases: the bias for each layer
    :param inputs: input values
    :param weights: weights of the layers
    :return:
    """


    z1 =  weights[0] @ inputs  + biases[0]
    x1 = sigmoid_function(z1)
    z2 = weights[1] @ x1 + biases[1]
    x2 = sigmoid_function(z2)
    z3 = weights[2] @ x2
    x3 = sigmoid_function(z3)
    return x3, z1, x1, z2, x2, z3


def forward_batch(input_batch: np.ndarray, weights: list[np.ndarray],
                  biases: list[np.ndarray]):
    z1s = []
    z2s = []
    z3s = []
    x1s = []
    x2s = []
    x3s = []
    for i, input_ in enumerate(input_batch):
        x3, z1, x1, z2, x2, z3 = forward_pass(
            input_,
            weights,
            biases
        )
        z1s.append(z1)
        z2s.append(z2)
        z3s.append(z3)
        x1s.append(x1)
        x2s.append(x2)
        x3s.append(x3)
    z1s = np.array(z1s)
    z2s = np.array(z2s)
    z3s = np.array(z3s)
    x1s = np.array(x1s)
    x2s = np.array(x2s)
    x3s = np.array(x3s)
    return x3s, z1s, x1s, z2s, x2s, z3s

def binary_cross_entropy_loss(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              eps = 1e-5):
    N = y_true.shape[0]
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -1/N * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
outputs = []
weights_list = [weights_matrix_0_1, weights_matrix_1_2, weights_matrix_2_3]
biases_list = [bias_1, bias_2]
for entry in x:
    output, z1, x1, z2, x2, z3 = forward_pass(entry, weights_list, biases_list)
    outputs.append(output)
outputs = np.array(outputs)
loss1 = binary_cross_entropy_loss(outputs, y)
print(loss1)

outputs, z1s, x1s, z2s, x2s, z3s = forward_batch(x, weights_list, biases_list)
loss2 = binary_cross_entropy_loss(outputs, y)
print(loss2)









