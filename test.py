import numpy as np

a = np.load(
    "outputs/pod/singular_values.npy"
)

b = np.load(
    "outputs/pod_snapshot/singular_values.npy"
)

print(
    np.allclose(
        a,
        b,
        rtol=1e-7,
        atol=1e-10,
    )
)
