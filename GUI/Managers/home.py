from kivy.uix.screenmanager import Screen
from kivy.core.window import Window


class HomeScreen(Screen):
    def on_touch_down(self, touch):
        self.manager.current = 'select'
        return super().on_touch_down(touch)

    def _on_keyboard_down(self, window, keycode, text, modifiers, inst):
        if keycode and len(keycode) == 2:
            if keycode[1]:
                self.manager.current = 'select'
                return True

        return False

    def on_enter(self, *args):
        self._keyboard = Window.bind(on_key_down=self._on_keyboard_down)

    def on_leave(self, *args):
        if self._keyboard:
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)