import os
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, ObjectProperty
from plyer import filechooser


class SelectScreen(Screen):
    selected_path = StringProperty("")
    processing_function = ObjectProperty(None)

    def open_native_file_chooser(self):
        try:
            selection = filechooser.open_file(
                title="Choisissez une image à traiter...",
                filters=[("Images", "*.png;*.jpg;*.jpeg")]
            )
            if selection:
                self.selected_path = selection[0]
        except Exception as e:
            print(f"Erreur lors de l'ouverture du sélecteur de fichiers : {e}")

    def on_process_button_press(self):
        if self.selected_path and os.path.exists(self.selected_path):
            if not self.processing_function:
                print("Erreur critique : La fonction de traitement n'a pas été fournie.")
                return

            output_image_path = self.processing_function(self.selected_path)

            result_screen = self.manager.get_screen('result')
            result_screen.input_image_path = self.selected_path
            result_screen.output_image_path = output_image_path

            self.manager.current = 'result'
        else:
            print("Veuillez sélectionner une image valide d'abord.")