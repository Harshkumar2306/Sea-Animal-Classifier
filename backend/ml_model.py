import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

class_names = [
    'Clams','Corals','Crabs','Dolphin','Eel','Fish','Jelly Fish','Lobster',
    'Nudibranchs','Octopus','Otter','Penguin','Puffers','Sea Rays','Sea Urchins',
    'Seahorse','Seal','Sharks','Shrimp','Squid','Starfish','Turtle_Tortoise','Whale'
]

class BioHMSC(nn.Module):
    def __init__(self, num_fine_classes=23, num_coarse_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnetv2_m",
            pretrained=False,
            num_classes=0
        )
        f = self.backbone.num_features
        self.shared_mlp = nn.Sequential(
            nn.Linear(f, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        self.species_head = nn.Linear(512, num_fine_classes)
        self.coarse_head = nn.Linear(512, num_coarse_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.shared_mlp(feats)
        return self.species_head(feats), self.coarse_head(feats)

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
MODEL_PATH = "BioHMSC_best_model.pth"

model = None

def get_model():
    global model
    if model is None:
        model = BioHMSC(num_fine_classes=len(class_names), num_coarse_classes=5)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict_image(img: Image.Image):
    m = get_model()
    m.eval() # Force eval mode right before prediction
    x = transform(img).unsqueeze(0).to(device)
    
    # Test-Time Augmentation (TTA) - Horizontal Flip
    x_flipped = torch.flip(x, dims=[3])
    
    with torch.no_grad():
        s_out_orig, _ = m(x)
        s_out_flipped, _ = m(x_flipped)
        outputs = (s_out_orig + s_out_flipped) / 2.0
        
        temperature = 0.7
        probs = torch.softmax(outputs / temperature, dim=1)
        
    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    
    results = []
    for i in range(3):
        results.append({
            "label": class_names[top3_indices[0][i].item()],
            "conf": top3_probs[0][i].item() * 100
        })
        
    return results
