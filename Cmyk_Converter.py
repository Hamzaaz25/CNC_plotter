import numpy as np
from PIL import Image
import os


class Cmyk:
    def __init__(self, image,height):
        self.image = image
        self.height = height



    def cmyk_paths(self):
        test = Image.open(self.image)
        test = test.resize((int(test.width * self.height / test.height), self.height))

        # --- Split into CMYK ---
        c, m, y, k = self.image_to_cmyk_parts(test)

        # --- Create an output folder ---
        output_dir = "TestingImages/CMYK_parts"
        os.makedirs(output_dir, exist_ok=True)
        test.save(os.path.join(output_dir, "test.png"))
        # --- Save each channel ---
        c.save(os.path.join(output_dir, "cyan.png"))
        m.save(os.path.join(output_dir, "magenta.png"))
        y.save(os.path.join(output_dir, "yellow.png"))
        k.save(os.path.join(output_dir, "black.png"))
        c_path = "TestingImages/CMYK_Parts/cyan.png"
        m_path = "TestingImages/CMYK_Parts/magenta.png"
        y_path = "TestingImages/CMYK_Parts/yellow.png"
        k_path = "TestingImages/CMYK_Parts/black.png"
        cmyk_layers = [c_path, m_path, y_path, k_path]

        return cmyk_layers

    def split_cmyk(self, rgb_array, threshhold=1):
        data = rgb_array.astype(float) / 255
        threshold = threshhold / 255

        channel_max = data.max(2)
        channel_max[channel_max < threshold] = threshold

        k = 1 - channel_max
        c = (1 - data[:, :, 0] - k) / channel_max
        m = (1 - data[:, :, 1] - k) / channel_max
        y = (1 - data[:, :, 2] - k) / channel_max

        result = 1 - np.array([c, m, y, k])
        return tuple([(_ * 255).round().astype(np.uint8) for _ in result])

    def image_to_cmyk_parts(self, image):
        data = np.asarray(image.convert("RGB"))
        cmyk = self.split_cmyk(data)
        return (Image.fromarray(_) for _ in cmyk)