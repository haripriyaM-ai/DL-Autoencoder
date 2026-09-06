# EXP-07 : DL- Convolutional Autoencoder for Image Denoising
### Name: HARI PRIYA M

### Register Number: 212224240047

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Image denoising is the process of removing noise from a corrupted image while preserving the important features of the original image. In this experiment, a convolutional autoencoder is developed to learn how to reconstruct clean images from noisy images.

The MNIST handwritten digit dataset is used for this experiment. The dataset contains grayscale images of handwritten digits with an image size of 28 × 28 pixels. Noise is artificially added to the input images, and the autoencoder is trained to reconstruct the original clean images.


## DESIGN STEPS

### STEP 1:

Load the MNIST dataset and convert the images into tensors using appropriate transformations.

### STEP 2:

Create training and testing DataLoaders for batch processing of the images.

### STEP 3:

Add random noise to the original images to create noisy input images for the denoising task.

### STEP 4:

Build a convolutional autoencoder consisting of an encoder to extract image features and a decoder to reconstruct the original image.

### STEP 5:

Train the autoencoder using Mean Squared Error loss and the Adam optimizer by comparing the reconstructed image with the original clean image.

### STEP 6:

Evaluate the trained model and visualize the original, noisy, and reconstructed images to observe the denoising performance.




## PROGRAM

### Name: HARI PRIYA M

### Register Number: 212224240047

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```python
# Initialize model
model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

```python
# Training function
def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, _ in loader:

            images = images.to(device)
            noisy_images = add_noise(images)

            optimizer.zero_grad()

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {epoch_loss:.6f}"
        )
```




### OUTPUT

### Model Summary
<img width="662" height="538" alt="image" src="https://github.com/user-attachments/assets/7407215a-8df8-4e1d-8925-6248f2b11bdf" />


### Training loss

## Original vs Noisy Vs Reconstructed Image
<img width="746" height="427" alt="image" src="https://github.com/user-attachments/assets/5491911a-5c00-4c1e-a2e3-5d73de81fab7" />

## RESULT

Thus, a convolutional autoencoder was successfully developed using PyTorch for image denoising. The model learned to remove noise from MNIST images and reconstruct images that are similar to the original clean images.
