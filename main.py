import pygame
import os
import sys
import cv2

UI_FOLDER = "ui elements"

class FullscreenUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Image GUI")
        self.clock = pygame.time.Clock()
        self.running = True

        self.images = {}  # Cache for loaded images

        self.current_screen = "main_menu"
        self.previous_screen = None

        self.load_images()
        self.main_loop()

    def load_image(self, filename):
        path = os.path.join(UI_FOLDER, filename)
        image = pygame.image.load(path).convert_alpha()
        return image

    def load_images(self):
        self.images['main'] = self.load_image("main.png")
        self.images['scan'] = self.load_image("scan.png")
        self.images['premodel'] = self.load_image("premodel.png")
        self.images['placeonscanner'] = self.load_image("placeonscanner.png")
        self.images['cancel'] = self.load_image("cancel_3.png")

        self.scan_rect = self.images['scan'].get_rect(topleft=(510, 100))
        self.premodel_rect = self.images['premodel'].get_rect(topleft=(165, 100))

        # cancel button will be centered later based on screen size
        screen_width, screen_height = self.screen.get_size()
        self.cancel_rect = self.images['cancel'].get_rect(center=(screen_width // 2, screen_height // 2))

    def draw_main_menu(self):
        self.screen.blit(self.images['main'], (0, 0))
        self.screen.blit(self.images['scan'], self.scan_rect.topleft)
        self.screen.blit(self.images['premodel'], self.premodel_rect.topleft)

    def draw_scanner_screen(self):
        self.screen.blit(self.images['placeonscanner'], (0, 0))

    def draw_white_screen_with_cancel(self):
        self.screen.fill((255, 255, 255))  # white background
        self.screen.blit(self.images['cancel'], self.cancel_rect.topleft)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.current_screen == "main_menu":
                    if self.scan_rect.collidepoint(event.pos):
                        self.run_camera_feeds()
                    elif self.premodel_rect.collidepoint(event.pos):
                        self.previous_screen = "main_menu"
                        self.current_screen = "white_cancel"

                elif self.current_screen == "white_cancel":
                    if self.cancel_rect.collidepoint(event.pos):
                        self.current_screen = self.previous_screen

    def run_camera_feeds(self):
        cap0 = cv2.VideoCapture(0)  # /dev/video0
        cap2 = cv2.VideoCapture(2)  # /dev/video2

        if not cap0.isOpened():
            print("Failed to open /dev/video0")
        if not cap2.isOpened():
            print("Failed to open /dev/video2")

        resize_factor = 0.2

        while True:
            ret0, frame0 = cap0.read()
            ret2, frame2 = cap2.read()

            if ret0:
                frame0_small = cv2.resize(frame0, (0, 0), fx=resize_factor, fy=resize_factor)
                cv2.imshow('Camera 0', frame0_small)

            if ret2:
                frame2_small = cv2.resize(frame2, (0, 0), fx=resize_factor, fy=resize_factor)
                cv2.imshow('Camera 2', frame2_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap0.release()
        cap2.release()
        cv2.destroyAllWindows()


    def main_loop(self):
        while self.running:
            self.handle_events()

            if self.current_screen == "main_menu":
                self.draw_main_menu()
            elif self.current_screen == "scanner":
                self.draw_scanner_screen()
            elif self.current_screen == "white_cancel":
                self.draw_white_screen_with_cancel()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    FullscreenUI()
