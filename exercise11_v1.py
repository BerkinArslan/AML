import numpy as np
import exersize10_solution as NN


x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  # inputs

y: np.ndarray = np.array([0, 1, 1, 0])  # targets

y_0 = y[0]
x_0 = x[0]
output, intermediates = NN.forward_single(x_0)
#print(intermediates)

y_pred_0 = output[0]


#derive the expression for the derivative of the MSE loss with respect to the output activation
# ∂L/∂ÿ = 2/n • (y - ÿ)

dldy = 2 * (y_pred_0 - y_0)
print(f'dldy: {dldy}')

#grads with respect to:
#1 outputlayer w and b
# ∂L/∂w_ij = ∂zi/∂w
# ∂L/∂w_ij = ∂zi/wij • ∂xi/∂zi • ∂L/∂xi
dzdw = intermediates[-3]
print(f'dzdw: {dzdw}')

dxdz = NN.sigmoid(intermediates[-2]) * (1 - NN.sigmoid(intermediates[-2]))
dxdz = intermediates[-1] * (1 - intermediates[-1])
print(f'dxdz: {dxdz}')

dldx = 2 * (intermediates[-1] - y_0)
print(f'dldx: {dldx}')

dldw = dzdw * dxdz * dldx
print(f'dldw: {dldw}')
