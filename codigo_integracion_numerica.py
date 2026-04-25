import numpy as np

PI = np.pi

def f(x):
    return 2*np.sqrt(np.maximum(0, 1 - np.asarray(x)**2))

def partition_equi(N):
    return np.linspace(-1, 1, N+1)

def partition_random(N, seed=12345):
    rng = np.random.default_rng(seed + N)
    return np.sort(np.concatenate(([-1, 1], rng.uniform(-1, 1, N-1))))

def partition_cos(N):
    i = np.arange(N+1)
    return np.sort(np.cos(i*np.pi/N))

def lower_upper_sum(x):
    lower = 0.0
    upper = 0.0
    for a, b in zip(x[:-1], x[1:]):
        fa, fb = f(a), f(b)
        h = b - a
        lower += min(fa, fb) * h
        upper += (2.0 if a <= 0 <= b else max(fa, fb)) * h
    return lower, upper

def rectangles_left(N):
    x = partition_equi(N)
    return np.sum(f(x[:-1]) * np.diff(x))

def trapezoid(N):
    x = partition_equi(N)
    return np.sum((f(x[:-1]) + f(x[1:]))/2 * np.diff(x))

def midpoint(N):
    x = partition_equi(N)
    m = (x[:-1] + x[1:])/2
    return np.sum(f(m) * np.diff(x))
