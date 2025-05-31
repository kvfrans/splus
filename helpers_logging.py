import matplotlib.pyplot as plt
import numpy as np

def create_horizontal_bars(data, labels, title, x_max=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, -np.array(data), align='center', color='skyblue')
    ax.barh(y_pos, np.array(data), align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.1, zorder=-1)
    if x_max is not None:
        ax.set_xlim(-x_max, x_max)
    return fig