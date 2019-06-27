import numpy as np

def get_efficient_port(S,E,c):
    Z = np.dot(np.linalg.inv(S), E)
    X = Z / np.sum(Z)
    Z = np.dot(np.linalg.inv(S), (E-c))
    Y = Z/np.sum(Z)
    E_X = np.dot(E.T,X)
    E_Y = np.dot(E.T,Y)
    var_X = np.dot(np.dot(X.T, S),X)
    var_Y = np.dot(np.dot(Y.T,S),Y)
    sigma_X = np.sqrt(var_X)
    sigma_Y = np.sqrt(var_Y)
    cov_XY = np.dot(np.dot(X.T, S),Y)
    corr_XY = cov_XY / (sigma_X*sigma_Y)
    def port_ret(alpha, E_X, E_Y):
        E_port = E_X*alpha + E_Y*(1-alpha)
        return E_port

    def port_vol(alpha, var_X, var_Y, cov_XY):
        var_port = np.sqrt(var_X*(alpha**2) + var_Y*((1-alpha)**2) + 2*alpha*(1-alpha)*cov_XY)
        return var_port
    alphas = np.arange(-15, 30, 1) * 0.1
    std_port = np.zeros(45)
    E_port = np.zeros(45)
    for i in range(45):
        alpha = alphas[i]
        std_port[i] = port_vol(alpha, var_X, var_Y, cov_XY)
        E_port[i] = port_ret(alpha, E_X, E_Y)

    sharp_ratio = (E_port - c) / std_port
    opt_ind = np.argmax(sharp_ratio)
    return E_port[opt_ind], std_port[opt_ind], alpha[opt_ind]
