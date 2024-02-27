
import torch 

def save_model(epochs, model, optimizer, criterion, path = "./best_model.pth"):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)
    print(f'Save model {epochs}')