from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.core.window import Window

Builder.load_file('GUI/Screens/styles.kv')
Builder.load_file('GUI/Screens/home.kv')
Builder.load_file('GUI/Screens/select.kv')
Builder.load_file('GUI/Screens/result.kv')

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