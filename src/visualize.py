import matplotlib.pyplot as plt


def visualize(X, y, m):
    plt.scatter(X, y)
    plt.title("HorsePower vs MPG")
    plt.plot(X, m.predict(X), color="red")
    plt.show()
