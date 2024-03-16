import matplotlib.pyplot as plt
from positional_encoding import positional_encoding
import tensorflow as tf

pos_encoding = positional_encoding(length=2048, depth=512)

# Check the shape.
print(pos_encoding.shape)

# Plot the dimensions.
plt.pcolormesh(pos_encoding.numpy().T, cmap="RdBu")
plt.ylabel("Depth")
plt.xlabel("Position")
plt.colorbar()
plt.show()


pos_encoding /= tf.norm(pos_encoding, axis=1, keepdims=True)
p = pos_encoding[1000]
dots = tf.einsum("pd,d->p", pos_encoding, p).numpy()

plt.subplot(2, 1, 1)
plt.plot(dots)
plt.ylim([0, 1])
plt.plot(
    [950, 950, float("nan"), 1050, 1050],
    [0, 1, float("nan"), 0, 1],
    color="k",
    label="Zoom",
)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(len(dots)), dots)
plt.xlim([950, 1050])
plt.ylim([0, 1])
