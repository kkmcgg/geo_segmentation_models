from scripts.main_thing import GeoDataGenerator, infer_on_image, save_mask

# Example: Fine-tuning on geospatial data
from segmentation_models import Unet
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    # Paths to geospatial images and masks
    train_images = ['path/to/train/image1.tif', 'path/to/train/image2.tif', ...]
    train_masks = ['path/to/train/mask1.tif', 'path/to/train/mask2.tif', ...]
    val_images = ['path/to/val/image1.tif', ...]
    val_masks = ['path/to/val/mask1.tif', ...]

    # Initialize data generators
    train_gen = GeoDataGenerator(train_images, train_masks, batch_size=16)
    val_gen = GeoDataGenerator(val_images, val_masks, batch_size=16)

    # Initialize and compile model
    model = Unet('resnet34', input_shape=(256, 256, 3), classes=1, activation='sigmoid')
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, validation_data=val_gen, epochs=50)

    # Perform inference
    mask, profile = infer_on_image(model, 'path/to/test/image.tif')
    save_mask(mask, profile, 'path/to/output/mask.tif')
