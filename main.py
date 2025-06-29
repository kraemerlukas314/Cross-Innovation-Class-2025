import pygame
import os
import sys
import cv2
import numpy as np

UI_FOLDER = "ui elements"

class FullscreenUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Image GUI")
        self.clock = pygame.time.Clock()
        self.running = True

        self.images = {}
        self.current_screen = "main_menu"
        self.previous_screen = None

        self.camera_active = False
        self.frame0 = None
        self.frame2 = None
        self.mask0 = None
        self.mask2 = None

        self.load_images()
        self.main_loop()

    def load_image(self, filename):
        path = os.path.join(UI_FOLDER, filename)
        return pygame.image.load(path).convert_alpha()

    def load_images(self):
        self.images['main'] = self.load_image("main.png")
        self.images['scan'] = self.load_image("scan.png")
        self.images['premodel'] = self.load_image("premodel.png")
        self.images['placeonscanner'] = self.load_image("placeonscanner.png")
        self.images['cancel'] = self.load_image("cancel_3.png")
        self.images['pause'] = self.load_image("pause.png")

        self.scan_rect = self.images['scan'].get_rect(topleft=(510, 100))
        self.premodel_rect = self.images['premodel'].get_rect(topleft=(165, 100))

        screen_width, screen_height = self.screen.get_size()
        self.cancel_rect = self.images['cancel'].get_rect(center=(screen_width // 2, screen_height // 2))
        self.pause_rect = self.images['pause'].get_rect(midbottom=(screen_width // 2, screen_height - 50))

    def draw_main_menu(self):
        self.screen.blit(self.images['main'], (0, 0))
        self.screen.blit(self.images['scan'], self.scan_rect.topleft)
        self.screen.blit(self.images['premodel'], self.premodel_rect.topleft)

    def draw_white_screen_with_cancel(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.images['cancel'], self.cancel_rect.topleft)

    def draw_camera_preview(self):
        self.screen.fill((255, 255, 255))
        if self.frame0 is not None:
            surf0 = self.cv_to_pygame(self.frame0)
            self.screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.frame2 is not None:
            surf2 = self.cv_to_pygame(self.frame2)
            self.screen.blit(surf2, surf2.get_rect(topright=(self.screen.get_width() - 50, 100)))
        self.screen.blit(self.images['pause'], self.pause_rect.topleft)

    def draw_mask_preview(self):
        self.screen.fill((255, 255, 255))
        if self.mask0 is not None:
            surf0 = self.gray_to_pygame(self.mask0)
            self.screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.mask2 is not None:
            surf2 = self.gray_to_pygame(self.mask2)
            self.screen.blit(surf2, surf2.get_rect(topright=(self.screen.get_width() - 50, 100)))
        self.screen.blit(self.images['pause'], self.pause_rect.topleft)

    def cv_to_pygame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (400, 300))
        return pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], 'RGB')

    def gray_to_pygame(self, gray_img):
        resized = cv2.resize(gray_img, (400, 300))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return pygame.image.frombuffer(rgb_img.tobytes(), rgb_img.shape[1::-1], 'RGB')

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.current_screen == "main_menu":
                    if self.scan_rect.collidepoint(event.pos):
                        self.start_camera_feeds()
                    elif self.premodel_rect.collidepoint(event.pos):
                        self.previous_screen = "main_menu"
                        self.current_screen = "white_cancel"

                elif self.current_screen == "white_cancel":
                    if self.cancel_rect.collidepoint(event.pos):
                        self.current_screen = self.previous_screen

                elif self.current_screen == "scanner":
                    if self.pause_rect.collidepoint(event.pos):
                        self.snap_image()

    def start_camera_feeds(self):
        self.cap0 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(2)
        if not self.cap0.isOpened():
            print("Failed to open /dev/video0")
        if not self.cap2.isOpened():
            print("Failed to open /dev/video2")

        self.camera_active = True
        self.current_screen = "scanner"

    def snap_image(self):
        if self.frame0 is not None and self.frame2 is not None:
            cv2.imwrite("snapshot_cam0.png", self.frame0)
            cv2.imwrite("snapshot_cam2.png", self.frame2)

            self.mask0 = self.process_mask(self.frame0)
            self.mask2 = self.process_mask(self.frame2)

            print("Snapped and created cleaned binary masks.")

            self.current_screen = "preview_mask"
            self.camera_active = False
            self.cap0.release()
            self.cap2.release()
            cv2.destroyAllWindows()

    def process_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.erode(binary, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(cleaned)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        return mask

    def update_camera_frames(self):
        if self.camera_active:
            ret0, frame0 = self.cap0.read()
            ret2, frame2 = self.cap2.read()
            if ret0:
                self.frame0 = frame0.copy()
            if ret2:
                self.frame2 = frame2.copy()

    def main_loop(self):
        while self.running:
            self.handle_events()
            self.update_camera_frames()

            if self.current_screen == "main_menu":
                self.draw_main_menu()
            elif self.current_screen == "white_cancel":
                self.draw_white_screen_with_cancel()
            elif self.current_screen == "scanner":
                self.draw_camera_preview()
            elif self.current_screen == "preview_mask":
                self.draw_mask_preview()

            pygame.display.flip()
            self.clock.tick(30)

        if hasattr(self, 'cap0'):
            self.cap0.release()
        if hasattr(self, 'cap2'):
            self.cap2.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    FullscreenUI()
