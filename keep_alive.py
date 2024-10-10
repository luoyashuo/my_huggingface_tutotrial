import numpy as np

memory_blocks = []
while True:
    try:
        memory_blocks.append(np.ones((1024, 1024, 1024), dtype=np.uint8))
    except MemoryError:
        del memory_blocks
        break