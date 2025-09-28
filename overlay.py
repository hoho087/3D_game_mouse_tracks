import pygame
import win32gui
import win32con
import ctypes
import threading
import queue
import time
from collections import deque

class OverlayDrawer:
    def __init__(self, max_shapes=600, refresh_rate=60):
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._drawing_thread, daemon=True)
        self.shapes = deque(maxlen=max_shapes)
        self.running = True
        self.refresh_rate = max(1, int(refresh_rate))

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        if not self.thread.is_alive():
            return
        self.update_queue.put(('clear',))
        self.update_queue.put(('stop',))
        self.stop_event.set()
        self.thread.join()

    def add_circle(self, center_x, center_y, radius, color, width):
        self.update_queue.put(('add_shape', ('circle', (center_x, center_y), radius, color, width)))

    def add_rect(self, x, y, x_width, y_height, color, width):
        self.update_queue.put(('add_shape', ('rect', color, (x, y, x_width, y_height), width)))

    def add_line(self, start_x, start_y, end_x, end_y, color, width):
        self.update_queue.put(('add_shape', ('line', (start_x, start_y), (end_x, end_y), color, width)))

    def clear(self):
        self.update_queue.put(('clear',))

    def _drawing_thread(self):
        pygame.init()
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.NOFRAME | pygame.SRCALPHA)

        hwnd = pygame.display.get_wm_info()["window"]

        extended_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        extended_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, extended_style)

        win32gui.SetLayeredWindowAttributes(hwnd, 0x000000, 255, win32con.LWA_COLORKEY)

        user32 = ctypes.windll.user32
        WDA_EXCLUDEFROMCAPTURE = 0x11
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

        def force_topmost():
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            if self.running:
                pygame.time.set_timer(pygame.USEREVENT, 100)

        clock = pygame.time.Clock()
        force_topmost()

        while not self.stop_event.is_set():
            try:
                while True:
                    task, *args = self.update_queue.get_nowait()
                    if task == 'clear':
                        self.shapes.clear()
                    elif task == 'add_shape':
                        shape = args[0]
                        self.shapes.append(shape)
                    elif task == 'stop':
                        self.running = False
            except queue.Empty:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.USEREVENT:
                    force_topmost()

            if not self.running:
                self.stop_event.set()

            screen.fill((0, 0, 0, 0))
            for shape in self.shapes:
                if shape[0] == 'circle':
                    _, center, radius, color, width = shape
                    pygame.draw.circle(screen, color, center, radius, width)
                elif shape[0] == 'line':
                    _, start, end, color, width = shape
                    pygame.draw.line(screen, color, start, end, width)
                elif shape[0] == 'rect':
                    _, color, rect, width = shape
                    pygame.draw.rect(screen, color, rect, width)

            pygame.display.flip()
            clock.tick(self.refresh_rate)

        pygame.quit()

if __name__ == "__main__":
    drawer = OverlayDrawer()
    drawer.start()

    drawer.add_line(0, 0, 1920, 1080, (0, 255, 0), 5)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        drawer.stop()