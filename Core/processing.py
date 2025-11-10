import os
import tempfile
import shutil
from PIL import Image
import torch
import numpy as np

def run_super_resolution(input_path: str, model, device="cuda" if torch.cuda.is_available() else "cpu") -> str:
    """
    Applique un modèle de super-résolution sur une image et renvoie le chemin de sortie.
    - input_path : chemin de l'image d'entrée
    - model : modèle PyTorch de super-résolution
    - device : 'cuda' ou 'cpu'
    """

    # Chargement de l'image
    img = Image.open(input_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Passage dans le modèle
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    # Conversion du tenseur de sortie en image
    output_img = output.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    output_img = Image.fromarray((output_img * 255).astype("uint8"))

    # Création du dossier temporaire et sauvegarde
    original_name = os.path.basename(input_path)
    name_part, ext_part = os.path.splitext(original_name)
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{name_part}_sr{ext_part}")

    output_img.save(output_path)

    return output_path