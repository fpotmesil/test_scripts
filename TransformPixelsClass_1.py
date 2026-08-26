import torch
import warnings
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", module="torchvision.datasets.cifar", message="dtype(): align should be passed as Python or NumPy boolean but got `align=0`")


'''
How It Works
PixelTranslate

Subclasses torchvision.transforms.v2.Transform so it integrates with the v2 API.
Uses torch.nn.functional.pad to shift pixels right/down, then crops back.
collate_with_translation

Receives the raw batch from the dataset.
Applies base_transform (convert to tensor, normalize).
Applies the pixel translation transform.
Stacks into a single tensor batch.
Why collate_fn?

Normally transforms are applied in the dataset’s __getitem__.
Using collate_fn lets you apply transforms after sampling, useful for batch-level or dynamic transforms.
✅ Edge Cases Handled

Non-tensor inputs are validated.
Works with any dataset returning (PIL.Image, label).
Keeps image size consistent after translation.
'''



# --- Custom pixel translation transform ---
class PixelTranslate(T.Transform):
    def __init__(self, shift_x: int, shift_y: int):
        super().__init__()
        self.shift_x = shift_x
        self.shift_y = shift_y

    def transform(self, inpt, params):
        # inpt is a tensor in (C, H, W)
        if not torch.is_tensor(inpt):
            raise TypeError("Expected input to be a torch.Tensor")
        c, h, w = inpt.shape
        # Pad and crop to shift pixels
        padded = F.pad(inpt, (self.shift_x, 0, self.shift_y, 0))  # pad left/top
        return padded[:, :h, :w]  # crop back to original size

# --- Define transforms ---
# v2 transforms work on both PIL and tensors
base_transform = T.Compose([
    T.ToImage(),           # Convert PIL -> Tensor
    T.ToDtype(torch.float32, scale=True),  # Normalize to [0,1]
])

translate_transform = PixelTranslate(shift_x=10, shift_y=3)

# --- Custom collate function ---
def collate_with_translation(batch):
    # batch is a list of (image, label) tuples
    images, labels = zip(*batch)
    images = [base_transform(img) for img in images]
    images = [translate_transform(img) for img in images]
    return torch.stack(images), torch.tensor(labels)

# --- Dataset & DataLoader ---
training_cifar10 = datasets.CIFAR10(root="./data", train=True, download=True)
train_images = training_cifar10.data
train_targets = training_cifar10.targets
print(f'train_targets len: {len(train_targets)}, set len: {len(set(train_targets))}')
## unique_training_values = train_targets.unique()
unique_training_values = len( set(train_targets) )

validation_cifar10 = datasets.CIFAR10(root="./data", train=False, download=True)
validation_images = validation_cifar10.data
validation_targets = validation_cifar10.targets


print(f'train_images & train_targets list len:\n\tX - {len(train_images)}\n\tY - {len(train_targets)}\n\t\
Y - Unique Values: {unique_training_values}')

print(f'TASK:\n\t {unique_training_values} class Classification')
print(f'UNIQUE CLASSES:\n\t {training_cifar10.classes}')

loader = DataLoader(training_cifar10, batch_size=8, shuffle=True,
        collate_fn=collate_with_translation)


# --- Example usage ---
if __name__ == "__main__":

    loops = 0

    for imgs, labels in loader:
        print("Batch images shape:", imgs.shape)  # (B, C, H, W)
        print("Batch labels:", labels)

        fig, ax = plt.subplots(1, 8, figsize=(10,10))
        for ix, axis in enumerate(ax.flat):
            axis.set_title(training_cifar10.classes[labels[ix]])
            img = imgs[ix]
            img2 = np.transpose(img, (1,2,0))
            axis.imshow(img2)
        plt.tight_layout()
        plt.show()

        loops += 1
        if loops > 4:
            break

