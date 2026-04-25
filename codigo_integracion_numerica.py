try:
    import numpy as np
    USE_NUMPY = True
except Exception:
    USE_NUMPY = False

if USE_NUMPY:
    PI = np.pi

    def f(x):
        x = np.asarray(x)
        return 2.0 * np.sqrt(np.maximum(0.0, 1.0 - x * x))

    def partition_equi(N):
        return np.linspace(-1.0, 1.0, N + 1)

    def partition_random(N, seed=12345):
        rng = np.random.default_rng(seed + N)
        return np.sort(np.concatenate(([-1.0, 1.0], rng.uniform(-1.0, 1.0, max(0, N - 1)))))

    def partition_cos(N):
        i = np.arange(N + 1)
        return np.sort(np.cos(i * np.pi / N))

    def lower_upper_sum(x):
        x = np.asarray(x)
        fa = f(x[:-1])
        fb = f(x[1:])
        h = np.diff(x)
        lower = np.sum(np.minimum(fa, fb) * h)
        mask = (x[:-1] <= 0.0) & (x[1:] >= 0.0)
        upper = np.sum(np.where(mask, 2.0, np.maximum(fa, fb)) * h)
        return float(lower), float(upper)

    def rectangles_left(N):
        x = partition_equi(N)
        return float(np.sum(f(x[:-1]) * np.diff(x)))

    def trapezoid(N):
        x = partition_equi(N)
        return float(np.sum((f(x[:-1]) + f(x[1:])) / 2.0 * np.diff(x)))

    def midpoint(N):
        x = partition_equi(N)
        m = (x[:-1] + x[1:]) / 2.0
        return float(np.sum(f(m) * np.diff(x)))

else:
    import math
    import random

    PI = math.pi

    def f(x):
        return 2.0 * math.sqrt(max(0.0, 1.0 - x * x))

    def partition_equi(N):
        return [-1.0 + i * (2.0 / N) for i in range(N + 1)]

    def partition_random(N, seed=12345):
        rng = random.Random(seed + N)
        pts = [-1.0, 1.0] + [rng.uniform(-1.0, 1.0) for _ in range(max(0, N - 1))]
        return sorted(pts)

    def partition_cos(N):
        return sorted([math.cos(i * math.pi / N) for i in range(N + 1)])

    def lower_upper_sum(x):
        lower = 0.0
        upper = 0.0
        for a, b in zip(x[:-1], x[1:]):
            fa, fb = f(a), f(b)
            h = b - a
            lower += min(fa, fb) * h
            if a <= 0.0 <= b:
                upper += 2.0 * h
            else:
                upper += max(fa, fb) * h
        return lower, upper

    def rectangles_left(N):
        x = partition_equi(N)
        s = 0.0
        for i in range(N):
            s += f(x[i]) * (x[i + 1] - x[i])
        return s

    def trapezoid(N):
        x = partition_equi(N)
        s = 0.0
        for i in range(N):
            s += (f(x[i]) + f(x[i + 1])) / 2.0 * (x[i + 1] - x[i])
        return s

    def midpoint(N):
        x = partition_equi(N)
        s = 0.0
        for i in range(N):
            m = (x[i] + x[i + 1]) / 2.0
            s += f(m) * (x[i + 1] - x[i])
        return s


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aproximaciones numéricas del área de la circunferencia unitaria")
    parser.add_argument("-N", type=int, default=1000, help="Número de subintervalos (por defecto 1000)")
    args = parser.parse_args()

    N = args.N
    print(f"Usando N = {N}")
    left = rectangles_left(N)
    trap = trapezoid(N)
    mid = midpoint(N)
    lower, upper = lower_upper_sum(partition_equi(N))

    print(f"Rectángulos (left): {left:.10f}, error = {abs(left-PI):.3e}")
    print(f"Trapecio:           {trap:.10f}, error = {abs(trap-PI):.3e}")
    print(f"Punto medio:        {mid:.10f}, error = {abs(mid-PI):.3e}")
    print(f"Cotas (lower,upper): {lower:.10f}, {upper:.10f}, errores = {abs(lower-PI):.3e}, {abs(upper-PI):.3e}")
    # Notificación final: mensaje y (si es Windows) sonido.
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass
    print("Ejecución completa.")

