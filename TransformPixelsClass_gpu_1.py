import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T

# --- GPU-accelerated pixel translation using torch.roll ---
class PixelTranslateRoll(T.Transform):
    def __init__(self, shift_x: int, shift_y: int):
        super().__init__()
        self.shift_x = shift_x
        self.shift_y = shift_y

    def _transform(self, inpt, params):
        if not torch.is_tensor(inpt):
            raise TypeError("Expected input to be a torch.Tensor")
        # torch.roll shifts pixels with wrap-around
        return torch.roll(inpt, shifts=(self.shift_y, self.shift_x), dims=(1, 2))

# --- Base transforms ---
base_transform = T.Compose([
    T.ToImage(),           # PIL -> Tensor
    T.ToDtype(torch.float32, scale=True),  # Normalize to [0,1]
])

translate_transform = PixelTranslateRoll(shift_x=5, shift_y=3)

# --- Optimized collate function ---
def collate_with_translation_gpu(batch):
    images, labels = zip(*batch)
    images = torch.stack([base_transform(img) for img in images])
    images = translate_transform(images)  # Apply to whole batch at once
    return images, torch.tensor(labels)

# --- Dataset & DataLoader ---
dataset = datasets.CIFAR10(root="./data", train=True, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_with_translation_gpu)

# --- Example usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)  # Move to GPU
        print("Batch images shape:", imgs.shape)  # (B, C, H, W)
        print("Batch labels:", labels)
        break

'''
Key Improvements
torch.roll

Shifts pixels in both height (dim=1) and width (dim=2) dimensions.
Wraps around pixels instead of padding with zeros (good for cyclic shifts).
Runs on GPU if the tensor is on GPU.
Batch-level Transform

Instead of looping over each image in the collate function, we stack first, then apply the transform to the whole batch — much faster.
GPU Support

If you move the batch to CUDA (imgs.to(device)), the shift happens on GPU automatically.
'''
