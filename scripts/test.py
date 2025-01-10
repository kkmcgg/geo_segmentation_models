import unittest
import numpy as np
from scripts.main_thing import tile_image, reconstruct_full_image

class TestGeoProcessing(unittest.TestCase):
    def test_tile_image(self):
        # Create a mock image
        image = np.random.rand(3, 512, 512)
        tiles = tile_image(image, tile_size=(256, 256))
        self.assertEqual(len(tiles), 4)

    def test_reconstruct_full_image(self):
        # Mock prediction
        predictions = np.random.rand(4, 256, 256, 1)
        original_shape = (512, 512)
        mask = reconstruct_full_image(predictions, original_shape, (256, 256), 0)
        self.assertEqual(mask.shape, original_shape)
