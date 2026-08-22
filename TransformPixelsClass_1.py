import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T
import torch.nn.functional as F

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

    def _transform(self, inpt, params):
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

translate_transform = PixelTranslate(shift_x=5, shift_y=3)

# --- Custom collate function ---
def collate_with_translation(batch):
    # batch is a list of (image, label) tuples
    images, labels = zip(*batch)
    images = [base_transform(img) for img in images]
    images = [translate_transform(img) for img in images]
    return torch.stack(images), torch.tensor(labels)

# --- Dataset & DataLoader ---
dataset = datasets.CIFAR10(root="./data", train=True, download=True)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_with_translation)

# --- Example usage ---
if __name__ == "__main__":
    for imgs, labels in loader:
        print("Batch images shape:", imgs.shape)  # (B, C, H, W)
        print("Batch labels:", labels)
        break

