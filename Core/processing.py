import os
import shutil
import tempfile


def run_super_resolution(input_path: str) -> str:
    #fonction principale qui gère le traitement de l'image
    #pour l'instant elle renvoie l'image de base

    original_name = os.path.basename(input_path)
    name_part, ext_part = os.path.splitext(original_name)

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{name_part}_processed{ext_part}")

    shutil.copy(input_path, output_path)

    return output_path