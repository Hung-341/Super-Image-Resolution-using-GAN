
# Image Super Resolution Project

This project implements a GAN-based Image Super Resolution model. The architecture is based on the combination of a Generator and a Discriminator that work together to produce high-resolution images from low-resolution inputs.

## Project Structure

- **Model Components:**
  - **Generator**: Upscales low-resolution images.
  - **Discriminator**: Distinguishes between real high-resolution images and generated super-resolution images.
  
- **Loss Functions**:
  - **Adversarial Loss (GAN)**: Used to train the generator and discriminator.
  - **VGG Loss (Perceptual Loss)**: Ensures perceptual quality by comparing feature maps of generated and real images from the VGG19 network.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hung-341/Image-Super-Resolution
   ```

3. Replace Kaggle directories with your local paths:
   - Note that the dataset paths in the code refer to Kaggle directories (`/kaggle/input/...`). You need to replace them with your own local dataset paths for training and validation data.

## Data Preprocessing

- **Low and High-Resolution Image Resizing**:
  - Low-resolution images are resized to 128x128, while high-resolution images are resized to 256x256 using the following transformations:
    ```python
    transform_low = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    transform_high = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    ```

- **Frequency Filtering**:
  - A frequency filter is applied to manipulate the image components in the frequency domain. You can use the `frequency_filter` function to apply low-pass or high-pass filters:
    ```python
    filtered_image = frequency_filter(image, cutoff_frequency=30, filter_type="lowpass")
    ```

## Model Architecture

- **Generator**:
  - The generator network follows a standard upscaling process using convolutional and upsampling blocks:
    - Residual Blocks for feature extraction.
    - Pixel Shuffle for upscaling.

- **Discriminator**:
  - The discriminator is a convolutional network designed to classify whether an image is real (high-resolution) or generated (super-resolution).

## Training Procedure

- **Training Loop**:
  - The model is trained using 60 epochs, with both generator and discriminator being updated at each iteration.
  - Checkpoints are saved every 40 epochs to prevent loss of training progress:
    ```python
    if i > 40:
        os.makedirs(savdir, exist_ok=True)
        torch.save(gen.state_dict(), savdir + "checkpoint1_gen")
        torch.save(disc.state_dict(), savdir + "checkpoint1_disc")
    ```

- **Histogram Matching**:
  - After generating the super-resolved image, you can match its histogram with the original low-resolution image to improve color consistency using the `with_histogram_matching` function:
    ```python
    matched_img = with_histogram_matching(upscaled_img, img_source)
    ```

## Results Visualization

- During training, the model outputs generated images, low-resolution inputs, and matched images using matplotlib for visualization:
  ```python
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
  ax1.imshow(generated_image)
  ax2.imshow(reference_image)
  ax3.imshow(matched_image)
  ```
  
## Project Contributors
- **Hung Le**
- **Khoa Le**
- **Hung Thinh**
- **Dung Nguyen**

## License

This project is licensed under the MIT License.
