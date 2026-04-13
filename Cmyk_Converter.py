import numpy as np
from PIL import Image
import os


class Cmyk:
    OUTPUT_DIR = "TestingImages/CMYK_Parts"

    def __init__(self, image, height):
        self.image = image
        self.height = height

    def cmyk_paths(self):
        img = Image.open(self.image)
        img = img.resize((int(img.width * self.height / img.height), self.height))

        c, m, y, k = self.image_to_cmyk_parts(img)

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        img.save(os.path.join(self.OUTPUT_DIR, "test.png"))

        channel_names = ["cyan", "magenta", "yellow", "black"]
        channels = [c, m, y, k]
        cmyk_layers = []

        for name, channel in zip(channel_names, channels):
            path = os.path.join(self.OUTPUT_DIR, f"{name}.png")
            channel.save(path)
            cmyk_layers.append(path)

        return cmyk_layers

    def split_cmyk(self, rgb_array, threshold=1):
        data = rgb_array.astype(float) / 255
        thresh = threshold / 255

        channel_max = data.max(2)
        channel_max[channel_max < thresh] = thresh

        k = 1 - channel_max
        c = (1 - data[:, :, 0] - k) / channel_max
        m = (1 - data[:, :, 1] - k) / channel_max
        y = (1 - data[:, :, 2] - k) / channel_max

        result = 1 - np.array([c, m, y, k])
        return tuple((ch * 255).round().astype(np.uint8) for ch in result)

    def image_to_cmyk_parts(self, image):
        data = np.asarray(image.convert("RGB"))
        cmyk = self.split_cmyk(data)
        return tuple(Image.fromarray(ch) for ch in cmyk)