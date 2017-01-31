from matplotlib import pyplot as plt
import numpy as np


class Plotter:

    @staticmethod
    def display(X, display_rows=10, display_cols=10, figsize=(4, 4), random_x=False):
        m = X.shape[0]
        fig, axes = plt.subplots(display_rows, display_cols, figsize=figsize)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        import random

        for i, ax in enumerate(axes.flat):
            ax.set_axis_off()
            x = None
            if random_x:
                x = random.randint(0, m-1)
            else:
                x = i
            image = X[x].reshape(20, 20).T
            image = image / np.max(image)
            ax.imshow(image, cmap=plt.cm.Greys_r)
        plt.show()
