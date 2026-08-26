import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T

# --- Zero-padded GPU pixel translation ---
class PixelTranslateZeroPad(T.Transform):
    def __init__(self, shift_x: int, shift_y: int):
        """
        shift_x: positive -> shift right, negative -> shift left
        shift_y: positive -> shift down, negative -> shift up
        """
        super().__init__()
        self.shift_x = shift_x
        self.shift_y = shift_y

    def _transform(self, inpt, params):
        if not torch.is_tensor(inpt):
            raise TypeError("Expected input to be a torch.Tensor")

        # Handle both single image (C,H,W) and batch (B,C,H,W)
        if inpt.ndim == 3:
            inpt = inpt.unsqueeze(0)  # Add batch dim

        b, c, h, w = inpt.shape
        out = torch.zeros_like(inpt)  # Zero-padded output

        # Compute source and destination slices
        src_x_start = max(0, -self.shift_x)
        src_x_end   = min(w, w - self.shift_x) if self.shift_x >= 0 else w
        dst_x_start = max(0, self.shift_x)
        dst_x_end   = dst_x_start + (src_x_end - src_x_start)

        src_y_start = max(0, -self.shift_y)
        src_y_end   = min(h, h - self.shift_y) if self.shift_y >= 0 else h
        dst_y_start = max(0, self.shift_y)
        dst_y_end   = dst_y_start + (src_y_end - src_y_start)

        # Copy shifted region
        out[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            inpt[:, :, src_y_start:src_y_end, src_x_start:src_x_end]

        return out if out.shape[0] > 1 else out.squeeze(0)  # Remove batch dim if needed

# --- Base transforms ---
base_transform = T.Compose([
    T.ToImage(),           # PIL -> Tensor
    T.ToDtype(torch.float32, scale=True),  # Normalize to [0,1]
])

translate_transform = PixelTranslateZeroPad(shift_x=5, shift_y=3)

# --- Collate function ---
def collate_with_translation_zero(batch):
    images, labels = zip(*batch)
    images = torch.stack([base_transform(img) for img in images])
    images = translate_transform(images)  # Apply to whole batch
    return images, torch.tensor(labels)

# --- Dataset & DataLoader ---
dataset = datasets.CIFAR10(root="./data", train=True, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_with_translation_zero)

# --- Example usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)  # GPU acceleration
        print("Batch images shape:", imgs.shape)  # (B, C, H, W)
        print("Batch labels:", labels)
        break

'''
How It Works
Zero Padding:

We create a zero tensor of the same shape as the input.
We compute the overlapping region between the source and destination after shifting.
We copy only the valid region, leaving the rest as zeros.
GPU Acceleration:

If the input batch is on CUDA, all operations (zeros_like, slicing, assignment) happen on GPU.
No Python loops over pixels — only tensor slicing.
Supports Batch or Single Image:

Works with (C, H, W) or (B, C, H, W) shapes.
'''
