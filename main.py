import sys
import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window

def resource_path(relative_path):
    #Pour avoir le chemin absolu vers les différentes ressources, nécessaire pour le .exe et .apk
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

Builder.load_file(resource_path('GUI/Screens/styles.kv'))
Builder.load_file(resource_path('GUI/Screens/home.kv'))
Builder.load_file(resource_path('GUI/Screens/select.kv'))
Builder.load_file(resource_path('GUI/Screens/result.kv'))

from GUI.Managers.home import HomeScreen
from GUI.Managers.select_screen import SelectScreen
from GUI.Managers.result_screen import ResultScreen
from Core.processing import run_super_resolution

Window.size = (800, 600)

class SuperResApp(App):
    def build(self):
        sm = ScreenManager()

        sm.add_widget(HomeScreen(name='home'))

        select_screen_widget = SelectScreen(
            name='select',
            processing_function=run_super_resolution
        )
        sm.add_widget(select_screen_widget)

        sm.add_widget(ResultScreen(name='result'))
        return sm

if __name__ == '__main__':
    SuperResApp().run()
