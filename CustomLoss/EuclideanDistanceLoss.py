import torch
import torch.nn as nn

class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, predictions, labels):
        # Ensure the input tensors are of shape [MINI_BATCH_SIZE, 3]
        assert predictions.shape == labels.shape, "Shape mismatch between predictions and labels"
        
        # Compute the squared differences between each coordinate (x, y, z)
        squared_diff = (predictions - labels) ** 2
        
        # Sum over the last dimension (x, y, z) to get the squared distance for each sample
        squared_distances = squared_diff.sum(dim=1)
        
        # Take the square root to get the Euclidean distance
        distances = torch.sqrt(squared_distances)
        
        # Return the mean distance over the mini-batch
        # return distances.mean()
        return distances.exp().mean()
        # return torch.exp(distances.mean())

if __name__ == "__main__":
    
    # Example usage
    predictions = torch.randn(8, 3)  # Assume mini-batch size is 8
    labels = torch.randn(8, 3)       # Same shape as predictions

    loss_fn = EuclideanDistanceLoss()
    loss = loss_fn(predictions, labels)
    print(loss)
