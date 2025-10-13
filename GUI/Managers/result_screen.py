import os
import shutil
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, BooleanProperty
from plyer import filechooser


class ResultScreen(Screen):
    input_image_path = StringProperty("")
    output_image_path = StringProperty("")
    download_complete = BooleanProperty(False)

    def download_image(self):
        #ouvre le filepicker pour enregistrer la nouvelle image
        if not self.output_image_path:
            print("Pas d'image à enregistrer.")
            return

        try:
            original_name = os.path.basename(self.output_image_path)
            name_part, ext_part = os.path.splitext(original_name)
            suggested_name = f"{name_part}_superres{ext_part}"

            save_path = filechooser.save_file(
                title="Enregistrer l'image sous...",
                default_path=suggested_name,
                filters=[("Image", f"*{ext_part}")]
            )

            if save_path:
                chosen_path = save_path[0]
                final_path = chosen_path

                if not os.path.splitext(chosen_path)[1]:
                    final_path += ext_part

                shutil.copy(self.output_image_path, final_path)
                print(f"Image enregistrée avec succès à : {final_path}")

                self.download_complete = True

        except Exception as e:
            print(f"Erreur lors de l'enregistrement du fichier : {e}")

    def back_to_select(self):
        self.manager.current = 'select'

    def on_leave(self, *args):
        self.download_complete = False
        super().on_leave(*args)