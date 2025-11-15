import os
from kivy.utils import platform
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty
from plyer import filechooser


class SelectScreen(Screen):
    selected_path = StringProperty("")
    image_source_path = StringProperty("")
    selected_model = StringProperty("")
    processing_function = ObjectProperty(None)

    def set_selected_model(self, model_name):
        if self.selected_model == model_name:
            self.selected_model = ""
        else:
            self.selected_model = model_name

    def open_native_file_chooser(self):
        try:
            filechooser.open_file(
                title="Choisissez une image à traiter...",
                filters=[("Images", "*.png;*.jpg;*.jpeg")],
                on_selection=self._handle_selection
            )
        except Exception as e:
            print(f"Erreur lors de l'ouverture du sélecteur de fichiers : {e}")

    def _handle_selection(self, selection):
        if not selection:
            self.selected_path = ""
            self.image_source_path = ""
            print("Sélection annulée.")
            return

        selected_path = selection[0]

        if selected_path is None:
            self.selected_path = ""
            self.image_source_path = ""
            print("Erreur de sélection (chemin None).")
            return

        if platform == 'android':
            self.selected_path = selected_path
            self.image_source_path = selected_path
            print(f"Image sélectionnée (Android) : {self.selected_path}")

        elif os.path.exists(selected_path):
            self.selected_path = selected_path
            self.image_source_path = self.selected_path
            print(f"Image sélectionnée (Desktop) : {self.selected_path}")

        else:
            self.selected_path = ""
            self.image_source_path = ""
            print(f"Erreur : chemin de fichier non valide : {selected_path}")

    def on_process_button_press(self):
        if self.selected_path and os.path.exists(self.selected_path) and self.selected_model:
            if not self.processing_function:
                print("Erreur critique : La fonction de traitement n'a pas été fournie.")
                return

            output_image_path = self.processing_function(self.selected_path)

            result_screen = self.manager.get_screen('result')
            result_screen.input_image_path = self.selected_path
            result_screen.output_image_path = output_image_path

            self.manager.current = 'result'
        else:
            if not self.selected_path:
                print("Veuillez sélectionner une image valide d'abord.")
            elif not self.selected_model:
                print("Veuillez sélectionner un modèle d'IA d'abord.")