import rasterio
import numpy as np

## Image tools
# could maybe just use tifffile instead of rasterio

def read_geospatial_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()  # Shape: (bands, height, width)
        profile = src.profile
    return image, profile

def tile_image(image, tile_size=(256, 256), overlap=0):
    bands, height, width = image.shape
    tile_h, tile_w = tile_size
    tiles = []
    for i in range(0, height, tile_h - overlap):
        for j in range(0, width, tile_w - overlap):
            tile = image[:, i:i+tile_h, j:j+tile_w]
            if tile.shape[1] == tile_h and tile.shape[2] == tile_w:
                tiles.append(tile)
    return np.array(tiles)


def reshape_for_model(tiles):
    # Assuming tiles are in (num_tiles, bands, height, width)
    reshaped = tiles.transpose(0, 2, 3, 1)  # Now (num_tiles, height, width, bands)
    return reshaped

def reconstruct_full_image(predictions, original_shape, tile_size, overlap):
    bands, height, width = original_shape
    tile_h, tile_w = tile_size
    full_mask = np.zeros((height, width))
    count_matrix = np.zeros((height, width))

    idx = 0
    for i in range(0, height, tile_h - overlap):
        for j in range(0, width, tile_w - overlap):
            if i + tile_h > height or j + tile_w > width:
                continue
            full_mask[i:i+tile_h, j:j+tile_w] += predictions[idx].squeeze()
            count_matrix[i:i+tile_h, j:j+tile_w] += 1
            idx += 1

    # Avoid division by zero
    count_matrix[count_matrix == 0] = 1
    full_mask /= count_matrix
    return full_mask

def save_mask(mask, profile, output_path): # could again maybe just use tifffile
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.float32), 1)


## model tools

from tensorflow.keras.utils import Sequence

class GeoDataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size=32, tile_size=(256, 256), shuffle=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.image_files[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        masks = []
        for img_path, mask_path in zip(batch_files, self.mask_files):
            img, _ = read_geospatial_image(img_path)
            mask, _ = read_geospatial_image(mask_path)
            img_tiles = tile_image(img, self.tile_size)
            mask_tiles = tile_image(mask, self.tile_size)
            images.append(reshape_for_model(img_tiles))
            masks.append(reshape_for_model(mask_tiles))
        images = np.vstack(images)
        masks = np.vstack(masks)
        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_files, self.mask_files))
            np.random.shuffle(combined)
            self.image_files, self.mask_files = zip(*combined)

if __name__ == '__main__':
    from segmentation_models import Unet
    from tensorflow.keras.optimizers import Adam

    train_image_files = ['train_image1.tif', 'train_image2.tif']
    train_mask_files = ['train_mask1.tif', 'train_mask2.tif']
    val_image_files = ['val_image1.tif', 'val_image2.tif']
    val_mask_files = ['val_mask1.tif', 'val_mask2.tif']

    for images, masks in train_gen:
        print(images.shape, masks.shape)
        break

    # Initialize model
    model = Unet('resnet34', input_shape=(256, 256, 3), classes=1, activation='sigmoid')
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Initialize data generators
    train_gen = GeoDataGenerator(train_image_files, train_mask_files, batch_size=16)
    val_gen = GeoDataGenerator(val_image_files, val_mask_files, batch_size=16)

    # Fine-tune the model
    model.fit(train_gen, validation_data=val_gen, epochs=50)


