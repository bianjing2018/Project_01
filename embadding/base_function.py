import numpy as np


def cosine_similar(vocter1, vocter2):
    up = np.dot(vocter1, vocter2)
    down = np.sqrt(np.sum(np.square(vocter1))) * np.sqrt(np.sum(np.square(vocter2)))
    return up / down
