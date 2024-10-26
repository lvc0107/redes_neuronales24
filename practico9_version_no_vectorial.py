import matplotlib.pyplot as plt
import numpy as np


def generate_cloud(num_points_per_class, sigma=0.1):
    range_cloud = 2
    center = (np.random.randint(range_cloud), np.random.randint(range_cloud))
    cloud = []
    for _ in range(num_points_per_class):
        x = center[0] + np.random.normal(scale=sigma)
        y = center[1] + np.random.normal(scale=sigma)
        new_point = (x, y)
        cloud.append(new_point)

    return cloud


num_points_per_class = 10
cloud_1 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_1), color="b")

cloud_2 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_2), color="g")

cloud_3 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_3), color="r")
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


x = np.array(cloud_1 + cloud_2 + cloud_3)


# expected results
y_classes = np.array(
    [(1, 0, 0) for _ in cloud_1]
    + [(0, 1, 0) for _ in cloud_2]
    + [(0, 0, 1) for _ in cloud_3]
)


def relu(h):
    return h if h > 0  else 0


def relu_derivative(h):
    return 1 if h > 0 else 0


def gradient_descent(x, y_classes, learning_rate=0.02, num_iterations=1000):
    num_samples, num_features = x.shape
    num_classes = y_classes.shape[1]

    # Set weights
    weights = np.random.randn(num_classes, num_features) * 0.01

    h = np.zeros((num_classes, num_samples))
    # Gradient descent iterations
    for _ in range(num_iterations):
        for m in range(num_samples):
            for j in range(num_classes):
                h[j, m] = 0.0
                for i in range(num_features):
                    h[j, m] += weights[j, i] * x[m, i]
                
        E = 0.0
        for m in range(num_samples):
            for p in range(num_classes):
                A = relu(h[p, m]) - y_classes[m, p]
                B = A * relu_derivative(h[p,m])
                E += A*A
                for q in range(num_features):
                    
                    weights[p, q] -= learning_rate *B * x[m, q]


    return weights


eta = 0.02
epoch = 1000

# Train model
weights = gradient_descent(x, y_classes, learning_rate=eta, num_iterations=epoch)
print(f"weights version escalar: {weights}")
print("weights")
# # Visualize dots and classes

# num_samples, num_features = weights.shape
# num_classes = y_classes.shape[1]

# def pred(weights, relu, x):
        
#     y = np.zeros(num_classes)
#     for j in range(num_classes):
#         h = 0.0
#         for i in range(num_features):
#             h += weights[j, i] * x[i]
        
#         y[j] = relu(h)
#     return y


# c = np.zeros(num_samples)
# for m in range(num_samples):
#     y = pred(weights, relu, x[m,:])
#     b = np.argmax(y)
#     c[m] = np.argmax(y_classes[m, :])
     
# plt.scatter(x[:, 0], x[:, 1], c=c, cmap="winter", alpha=0.5)
# plt.title("Dots clasificated with ReLU model")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
