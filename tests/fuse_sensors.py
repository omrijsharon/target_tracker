import numpy as np
import matplotlib.pyplot as plt


def distance_matrix(p):
    """Compute the distance matrix between points in p."""
    return np.sqrt(((p[:, None] - p[None, :]) ** 2).sum(axis=2))


def find_weights_with_constrain(distances, real_distance, lr=0.1, max_iter=1000, eps=1e-6):
    #initial weights:
    w = np.ones_like(distances) / len(distances)
    #loss function:
    def loss_func(w):
        return (w @ distances - real_distance) ** 2

    #gradient of loss function:
    def grad(w):
        return 2 * (w @ distances - real_distance) * distances

    def hessian_matrix(w):
        return 2 * distances * distances.T

    def newton_raphson_step(w):
        return w - np.linalg.inv(hessian_matrix(w)) @ grad(w)

    #gradient descent:
    for i in range(max_iter):
        w -= lr * grad(w)
        # w = newton_raphson_step(w)
        w = np.clip(w, 0, 1)
        w /= w.sum()
        loss = loss_func(w)
        print(f"{i} : {loss}")
        if loss < eps:
            break
    return w


if __name__ == '__main__':
    N = 4
    noise_level = 0.5
    n = N * (N - 1)
    t = np.linspace(0, 2*np.pi, N+1)[:-1]
    radius = 1
    p = np.array([radius * np.cos(t), radius * np.sin(t)]).T
    # plt.scatter(p[:, 0], p[:, 1])
    # plt.show()
    d_org = distance_matrix(p)
    noise_matrix = np.random.randn(N, N)
    noise_matrix = noise_level * (noise_matrix + noise_matrix.T) / 2 * (1-np.eye(N))
    d0 = d_org + noise_matrix
    noise_matrix = np.random.randn(N, N)
    d1 = d_org + noise_matrix



    # d0_org= np.linalg.norm(p, axis=1)
    # d0 = d0_org + noise_level * np.random.randn(N)
    # d1 = d0_org + noise_level * np.random.randn(N)
    # w0 = find_weights_with_constrain(d0, radius, lr=0.05, max_iter=1000, eps=1e-6)
    # print(w0)

