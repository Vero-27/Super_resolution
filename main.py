import sys
import os
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.core.text import LabelBase

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


Builder.load_file(resource_path('GUI/Screens/styles.kv'))
Builder.load_file(resource_path('GUI/Screens/home.kv'))
Builder.load_file(resource_path('GUI/Screens/select.kv'))
Builder.load_file(resource_path('GUI/Screens/result.kv'))

LabelBase.register(
    name='MaterialIcons',
    fn_regular=resource_path('fonts/MaterialIcons-Regular.ttf')
)

from GUI.Managers.home import HomeScreen
from GUI.Managers.select_screen import SelectScreen
from GUI.Managers.result_screen import ResultScreen
from Core.processing import run_super_resolution


class SuperResApp(App):
    def build(self):

        if platform == 'android':
            print("Vérification/Demande des permissions de stockage...")
            permissions = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
            request_permissions(permissions, self.on_permissions_granted)

        sm = ScreenManager()

        sm.add_widget(HomeScreen(name='home'))

        select_screen_widget = SelectScreen(
            name='select',
            processing_function=run_super_resolution
        )
        sm.add_widget(select_screen_widget)

        sm.add_widget(ResultScreen(name='result'))
        return sm

    def on_permissions_granted(self, permissions, grants):
        if all(grants):
            print("Permissions de stockage accordées !")
        else:
            print("Permissions de stockage refusées ! L'application risque de ne pas fonctionner.")


if __name__ == '__main__':
    SuperResApp().run()