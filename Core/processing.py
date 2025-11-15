import os
import tempfile
from shutil import copyfile

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Import des 3 modèles
from Core.models.espcn import ESPCN
from Core.models.srcnn import SRCNN
from Core.models.edsr_lite import EDSRLite

# Permet de decider si on fait tourner le pgrm sur la cg ou le processeur
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chemins vers les ficheirs de poids
ESPCN_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "espcn_x2.pth")
SRCNN_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "srcnn_x2.pth")
EDSR_CKPT  = os.path.join(os.path.dirname(__file__), "checkpoints", "edsr_lite_x2.pth")

ESPCN_MODEL = None
SRCNN_MODEL = None
EDSR_MODEL  = None



def _create_temp_output_path(input_path: str, suffix: str = "_superres") -> str:
    original_name = os.path.basename(input_path)
    name_part, ext_part = os.path.splitext(original_name)

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{name_part}{suffix}{ext_part}")
    return output_path


@torch.inference_mode()
def run_super_resolution_espcn(input_path: str) -> str:
    global ESPCN_MODEL

    # 1) Charger le modèle une seule fois
    if ESPCN_MODEL is None:
        if not os.path.exists(ESPCN_CKPT):
            print(f"[ESPCN] Checkpoint introuvable : {ESPCN_CKPT}")
            
            output_path = _create_temp_output_path(input_path, "_espcn_raw")
            copyfile(input_path, output_path)
            return output_path

        ckpt = torch.load(ESPCN_CKPT, map_location=DEVICE)
        scale = ckpt.get("scale", 2)

        model = ESPCN(scale=scale)
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        model.eval()

        ESPCN_MODEL = model
        print(f"[ESPCN] Modèle chargé (x{scale}) sur {DEVICE}")

    # 2) Charger l'image
    img = Image.open(input_path).convert("RGB")
    lr = TF.to_tensor(img).unsqueeze(0).to(DEVICE) 

    # 3) Appliquer le modèle
    sr = ESPCN_MODEL(lr).clamp(0, 1)

    # 4) Sauvegarder le résultat
    sr_img = TF.to_pil_image(sr.squeeze(0).cpu())
    output_path = _create_temp_output_path(input_path, "_espcn")
    sr_img.save(output_path)

    return output_path


@torch.inference_mode()
def run_super_resolution_srcnn(input_path: str) -> str:
    global SRCNN_MODEL

    if SRCNN_MODEL is None:
        if not os.path.exists(SRCNN_CKPT):
            print(f"[SRCNN] Checkpoint introuvable : {SRCNN_CKPT}")
            output_path = _create_temp_output_path(input_path, "_srcnn_raw")
            copyfile(input_path, output_path)
            return output_path

        ckpt = torch.load(SRCNN_CKPT, map_location=DEVICE)
        model = SRCNN()
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        model.eval()

        SRCNN_MODEL = model
        print(f"[SRCNN] Modèle chargé sur {DEVICE}")

    img = Image.open(input_path).convert("RGB")

    lr = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
    lr_up = torch.nn.functional.interpolate(
        lr, scale_factor=2, mode="bicubic", align_corners=False
    )

    sr = SRCNN_MODEL(lr_up).clamp(0, 1)

    sr_img = TF.to_pil_image(sr.squeeze(0).cpu())
    output_path = _create_temp_output_path(input_path, "_srcnn")
    sr_img.save(output_path)

    return output_path


@torch.inference_mode()
def run_super_resolution_edsr_lite(input_path: str) -> str:
    global EDSR_MODEL

    if EDSR_MODEL is None:
        if not os.path.exists(EDSR_CKPT):
            print(f"[EDSR] Checkpoint introuvable : {EDSR_CKPT}")
            output_path = _create_temp_output_path(input_path, "_edsr_raw")
            copyfile(input_path, output_path)
            return output_path

        ckpt = torch.load(EDSR_CKPT, map_location=DEVICE)
        scale = ckpt.get("scale", 2)

        model = EDSRLite(scale=scale)
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        model.eval()

        EDSR_MODEL = model
        print(f"[EDSR] Modèle EDSR-lite chargé (x{scale}) sur {DEVICE}")

    img = Image.open(input_path).convert("RGB")
    lr = TF.to_tensor(img).unsqueeze(0).to(DEVICE)

    sr = EDSR_MODEL(lr).clamp(0, 1)

    sr_img = TF.to_pil_image(sr.squeeze(0).cpu())
    output_path = _create_temp_output_path(input_path, "_edsr")
    sr_img.save(output_path)

    return output_path

