import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import cv2
import numpy as np
import io
import pygame
from pi5neo import Pi5Neo
import time
from stl import mesh
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

# --- Constants for Configuration ---

# Set to True to rotate the display 180 degrees, False for no rotation.
ROTATE_SCREEN = True

# Adjust the color detection sensitivity for creating masks.
# Lower values are more sensitive to darker colors. Range: 0-255.
COLOR_THRESHOLD = 60

# Adjust the brightness of the LED strip.
# Value is a percentage (0.0 to 1.0). 0.3 means 30% brightness.
LED_BRIGHTNESS = 0.3
# --- End of Constants ---


# Initialize the Pi5Neo class with 144 LEDs and an SPI speed of 800kHz
neo = Pi5Neo('/dev/spidev0.0', 144, 800)

UI_FOLDER = "ui elements"

class FullscreenUI:
    def __init__(self):
        pygame.init()
        # Set up the main display screen
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        
        # Create a virtual screen for potential rotation
        self.virtual_screen = pygame.Surface((self.screen_width, self.screen_height))
        
        pygame.display.set_caption("Image GUI")
        self.clock = pygame.time.Clock()
        self.running = True

        self.images = {}
        self.current_screen = "main_menu"
        self.previous_screen = None

        self.frame0 = None
        self.frame2 = None
        self.mask0 = None
        self.mask2 = None
        self.normalized_mask0 = None
        self.normalized_mask2 = None
        self.stl_preview_surface = None
        self.progress = 0
        
        # Calculate LED brightness value from the constant
        self.led_value = int(255 * LED_BRIGHTNESS)

        # Initialize cameras on startup
        self.cap0 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(2)
        if not self.cap0.isOpened():
            print("Failed to open /dev/video0")
        if not self.cap2.isOpened():
            print("Failed to open /dev/video2")

        # Ensure LEDs are off at the start
        neo.fill_strip(0, 0, 0)
        neo.update_strip()

        self.load_images()
        self.main_loop()

    def load_image(self, filename):
        """Loads an image from the UI folder."""
        path = os.path.join(UI_FOLDER, filename)
        return pygame.image.load(path).convert_alpha()

    def load_images(self):
        """Loads all UI images and defines their positions."""
        self.images['main'] = self.load_image("main.png")
        self.images['scan'] = self.load_image("scan.png")
        self.images['premodel'] = self.load_image("premodel.png")
        self.images['placeonscanner'] = self.load_image("placeonscanner.png")
        self.images['cancel'] = self.load_image("cancel_3.png")
        self.images['next'] = self.load_image("next_orange.png")
        self.images['try_again'] = self.load_image("try_again.png")
        self.images['package_design'] = self.load_image("start packaging design_orange.png")

        self.scan_rect = self.images['scan'].get_rect(topleft=(510, 100))
        self.premodel_rect = self.images['premodel'].get_rect(topleft=(165, 100))
        self.cancel_rect = self.images['cancel'].get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        
        # --- Change: Center buttons below their respective preview windows ---
        # Preview windows are 400x300. Left one starts at x=50, right one ends at screen_width-50
        # Calculate center x for each preview window
        left_preview_center_x = 50 + 400 // 2
        right_preview_center_x = (self.screen_width - 50) - 400 // 2
        
        # Y position for buttons below the previews (previews are at y=100, height=300)
        button_y_pos = 100 + 300 + 80 # 60 pixels below the previews
        
        # "Try Again" button centered under the left preview
        self.try_again_rect = self.images['try_again'].get_rect(center=(left_preview_center_x, button_y_pos))
        
        # "next/Continue" button centered under the right preview
        self.next_rect = self.images['next'].get_rect(center=(right_preview_center_x, button_y_pos))

        # --- Change: Adjust button positions on the final STL preview screen ---
        # Move "Try Again" further to the left
        self.stl_try_again_rect = self.images['try_again'].get_rect(bottomleft=(20, self.screen_height - 30))
        
        # Move "Start Packaging Design" further to the right
        self.package_rect = self.images['package_design'].get_rect(bottomright=(self.screen_width - 20, self.screen_height - 30))


    def draw_main_menu(self):
        """Draws the main menu screen."""
        self.virtual_screen.fill((0,0,0)) 
        self.virtual_screen.blit(self.images['main'], (0, 0))
        self.virtual_screen.blit(self.images['scan'], self.scan_rect.topleft)
        self.virtual_screen.blit(self.images['premodel'], self.premodel_rect.topleft)

    def draw_white_screen_with_cancel(self):
        """Draws a white screen with a cancel button."""
        self.virtual_screen.fill((255, 255, 255))
        self.virtual_screen.blit(self.images['cancel'], self.cancel_rect.topleft)

    def draw_camera_preview(self):
        """Draws the live camera feeds."""
        self.virtual_screen.fill((255, 255, 255))
        # Turn LEDs on for preview using the brightness constant
        neo.fill_strip(self.led_value, self.led_value, self.led_value)
        neo.update_strip()
        
        if self.frame0 is not None:
            surf0 = self.cv_to_pygame(self.frame0)
            self.virtual_screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.frame2 is not None:
            surf2 = self.cv_to_pygame(self.frame2)
            self.virtual_screen.blit(surf2, surf2.get_rect(topright=(self.virtual_screen.get_width() - 50, 100)))
            
        # Draw buttons in their new positions
        self.virtual_screen.blit(self.images['try_again'], self.try_again_rect.topleft)
        self.virtual_screen.blit(self.images['next'], self.next_rect.topleft)

    def draw_mask_preview(self):
        """Draws the generated binary masks."""
        self.virtual_screen.fill((255, 255, 255))
        if self.mask0 is not None:
            surf0 = self.gray_to_pygame(self.mask0)
            self.virtual_screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.mask2 is not None:
            surf2 = self.gray_to_pygame(self.mask2)
            self.virtual_screen.blit(surf2, surf2.get_rect(topright=(self.virtual_screen.get_width() - 50, 100)))
            
        # Draw buttons in their new positions
        self.virtual_screen.blit(self.images['try_again'], self.try_again_rect.topleft)
        self.virtual_screen.blit(self.images['next'], self.next_rect.topleft)

    def draw_normalized_preview(self):
        """Draws the final, normalized masks before 3D generation."""
        self.virtual_screen.fill((255, 255, 255))
        if self.normalized_mask0 is not None:
            surf0 = self.gray_to_pygame(self.normalized_mask0)
            self.virtual_screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.normalized_mask2 is not None:
            surf2 = self.gray_to_pygame(self.normalized_mask2)
            self.virtual_screen.blit(surf2, surf2.get_rect(topright=(self.virtual_screen.get_width() - 50, 100)))
            
        # Draw buttons in their new positions
        self.virtual_screen.blit(self.images['try_again'], self.try_again_rect.topleft)
        self.virtual_screen.blit(self.images['next'], self.next_rect.topleft)

    def draw_generating_stl_screen(self):
        """Draws the progress bar screen."""
        self.virtual_screen.fill((255, 255, 255))
        font = pygame.font.Font(None, 50)
        text = font.render("Generating 3D Model...", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.virtual_screen.blit(text, text_rect)
        self.draw_progress_bar(self.progress)

    def draw_stl_preview_screen(self):
        """Draws the rendered STL preview."""
        self.virtual_screen.fill((255, 255, 255))
        if self.stl_preview_surface:
            preview_rect = self.stl_preview_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
            
            # Draw a rounded rectangle around the preview
            border_rect = preview_rect.inflate(20, 20)
            pygame.draw.rect(self.virtual_screen, (220, 220, 220), border_rect, border_radius=15)
            
            # Draw the STL preview on top of the rounded rectangle
            self.virtual_screen.blit(self.stl_preview_surface, preview_rect)
            
        # Use the specially positioned rects for the STL screen
        self.virtual_screen.blit(self.images['try_again'], self.stl_try_again_rect.topleft)
        self.virtual_screen.blit(self.images['package_design'], self.package_rect.topleft)

    def draw_progress_bar(self, progress):
        """Draws a rounded, orange progress bar."""
        bar_width = 400
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = self.screen_height // 2
        
        bg_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.virtual_screen, (200, 200, 200), bg_rect, border_radius=15)

        progress_width = int(bar_width * progress)
        if progress_width > 0:
            progress_rect = pygame.Rect(bar_x, bar_y, progress_width, bar_height)
            pygame.draw.rect(self.virtual_screen, (255, 140, 0), progress_rect, border_radius=15)

    def cv_to_pygame(self, frame):
        """Converts an OpenCV (BGR) image to a Pygame surface."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (400, 300))
        return pygame.image.frombuffer(frame_rgb.tobytes(), frame_rgb.shape[1::-1], 'RGB')

    def gray_to_pygame(self, gray_img):
        """Converts an OpenCV grayscale image to a Pygame surface."""
        resized = cv2.resize(gray_img, (400, 300))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return pygame.image.frombuffer(rgb_img.tobytes(), rgb_img.shape[1::-1], 'RGB')
    
    def get_transformed_mouse_pos(self, pos):
        """Transforms mouse coordinates if screen rotation is enabled."""
        if ROTATE_SCREEN:
            return (self.screen_width - pos[0], self.screen_height - pos[1])
        else:
            return pos

    def handle_events(self):
        """Handles all user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Transform mouse position for collision detection based on rotation constant
                mouse_pos = self.get_transformed_mouse_pos(event.pos)

                if self.current_screen == "main_menu":
                    if self.scan_rect.collidepoint(mouse_pos):
                        self.start_camera_feeds()
                    elif self.premodel_rect.collidepoint(mouse_pos):
                        self.previous_screen = "main_menu"
                        self.current_screen = "white_cancel"

                elif self.current_screen == "white_cancel":
                    if self.cancel_rect.collidepoint(mouse_pos):
                        self.current_screen = self.previous_screen

                # --- Change: Use correct rects for collision on preview screens ---
                elif self.current_screen in ["scanner", "preview_mask", "normalized_mask"]:
                    if self.next_rect.collidepoint(mouse_pos):
                        # Action for the right button (next/continue)
                        if self.current_screen == "scanner":
                            self.snap_image()
                        elif self.current_screen == "preview_mask":
                            self.normalized_mask0 = self.normalize_mask(self.mask0)
                            self.normalized_mask2 = self.normalize_mask(self.mask2)
                            self.current_screen = "normalized_mask"
                        elif self.current_screen == "normalized_mask":
                            self.current_screen = "generating_stl"
                            
                    elif self.try_again_rect.collidepoint(mouse_pos):
                        # Action for the left button (try again/back)
                        self.start_camera_feeds()
                
                elif self.current_screen == "preview_stl":
                    if self.package_rect.collidepoint(mouse_pos):
                        self.current_screen = "main_menu"
                    # Use the special rect for this screen's 'try again' button
                    elif self.stl_try_again_rect.collidepoint(mouse_pos):
                        self.start_camera_feeds()

    def start_camera_feeds(self):
        """Sets the screen to the camera preview."""
        self.current_screen = "scanner"

    def snap_image(self):
        """Captures images from cameras and generates initial masks."""
        if self.frame0 is not None and self.frame2 is not None:
            cv2.imwrite("snapshot_cam0.png", self.frame0)
            cv2.imwrite("snapshot_cam2.png", self.frame2)

            self.mask0 = self.process_mask(self.frame0)
            self.mask2 = self.process_mask(self.frame2)

            print("Snapped and created cleaned binary masks.")
            self.current_screen = "preview_mask"

    def process_mask(self, image):
        """Converts an image to a clean, filled, binary mask of the largest object."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Use the color threshold constant
        _, binary = cv2.threshold(blurred, COLOR_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.erode(binary, kernel, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(cleaned)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

        return mask

    def normalize_mask(self, mask):
        """Rotates, crops, and scales a mask to a standard size and position."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90

        (h, w) = mask.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cropped = rotated[y:y+h, x:x+w]

        TARGET_WIDTH = 300
        scale = TARGET_WIDTH / w
        new_size = (TARGET_WIDTH, int(h * scale))
        resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((400, 300), dtype=np.uint8)
        y_offset = (canvas.shape[0] - resized.shape[0]) // 2
        x_offset = (canvas.shape[1] - resized.shape[1]) // 2
        canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

        return canvas

    def generate_stl_and_preview(self):
        """
        Generates a 3D model, saves it, and creates a preview image,
        updating a progress bar throughout the process.
        """
        if self.normalized_mask0 is None or self.normalized_mask2 is None:
            print("No normalized masks available to generate STL.")
            self.current_screen = "main_menu"
            return
        
        # Stage 1: Initialization
        self.progress = 0.1
        self.draw_generating_stl_screen()
        self.render_and_flip_screen() # Update display

        z_dim, y_dim, x_dim = 30, 30, 30
        side_mask_resized = cv2.resize(self.normalized_mask0, (x_dim, y_dim), interpolation=cv2.INTER_NEAREST)
        top_mask_resized = cv2.resize(self.normalized_mask2, (x_dim, z_dim), interpolation=cv2.INTER_NEAREST)
        
        # Stage 2: Voxel Carving
        self.progress = 0.25
        self.draw_generating_stl_screen()
        self.render_and_flip_screen() # Update display

        side_silhouette = side_mask_resized > 128
        top_silhouette = top_mask_resized > 128
        volume_from_side = np.broadcast_to(side_silhouette[np.newaxis, :, :], (z_dim, y_dim, x_dim))
        volume_from_top = np.broadcast_to(top_silhouette[:, np.newaxis, :], (z_dim, y_dim, x_dim))
        volume = np.logical_and(volume_from_side, volume_from_top)
        volume = np.pad(volume, 1, constant_values=False)
        
        # Stage 3: Marching Cubes
        self.progress = 0.35
        self.draw_generating_stl_screen()
        self.render_and_flip_screen() # Update display

        spacing_z, spacing_y, spacing_x = 400.0/z_dim, 400.0/y_dim, 300.0/x_dim
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(spacing_z, spacing_y, spacing_x))
        self.progress = 0.45
        
        # Stage 4: Creating STL Mesh
        self.progress = 0.75
        self.draw_generating_stl_screen()
        self.render_and_flip_screen() # Update display

        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            stl_mesh.vectors[i] = verts[f]
        stl_mesh.save('/home/pi/repo/model.stl')
        print("STL file saved as 'model.stl'")
        
        # Stage 5: Rendering Preview
        self.progress = 0.9
        self.draw_generating_stl_screen()
        self.render_and_flip_screen() # Update display

        fig = plt.figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        face_color_tuple = (160/255, 160/255, 160/255)
        edge_color_tuple = (0.2, 0.2, 0.2)
        collection = art3d.Poly3DCollection(
            stl_mesh.vectors, 
            facecolors=face_color_tuple, 
            edgecolors=edge_color_tuple,
            linewidths=0.5
        )
        ax.add_collection3d(collection)

        scale = stl_mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        ax.view_init(elev=30, azim=45)
        ax.set_axis_off()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        self.stl_preview_surface = pygame.image.load(buf).convert_alpha()
        plt.close(fig)

        self.progress = 1.0
        self.current_screen = "preview_stl"

    def update_camera_frames(self):
        """Reads the latest frames from the active cameras."""
        if self.cap0.isOpened():
            ret0, frame0 = self.cap0.read()
            if ret0:
                self.frame0 = frame0.copy()
        if self.cap2.isOpened():
            ret2, frame2 = self.cap2.read()
            if ret2:
                self.frame2 = frame2.copy()

    def render_and_flip_screen(self):
        """Rotates the virtual screen (if enabled) and updates the main display."""
        if ROTATE_SCREEN:
            rotated_surface = pygame.transform.rotate(self.virtual_screen, 180)
            self.screen.blit(rotated_surface, (0, 0))
        else:
            self.screen.blit(self.virtual_screen, (0, 0))
        pygame.display.flip()

    def main_loop(self):
        """The main application loop."""
        while self.running:
            self.handle_events()
            
            # This logic now runs continuously in the background
            self.update_camera_frames()

            # Turn off LEDs by default, turn on only where needed
            if self.current_screen != "scanner":
                neo.fill_strip(0, 0, 0)
                neo.update_strip()

            # --- Screen Drawing (on virtual_screen) ---
            if self.current_screen == "main_menu":
                self.draw_main_menu()
            elif self.current_screen == "white_cancel":
                self.draw_white_screen_with_cancel()
            elif self.current_screen == "scanner":
                self.draw_camera_preview()
            elif self.current_screen == "preview_mask":
                self.draw_mask_preview()
            elif self.current_screen == "normalized_mask":
                self.draw_normalized_preview()
            elif self.current_screen == "generating_stl":
                # The generation function handles its own drawing and flipping
                self.generate_stl_and_preview()
                continue # Skip the main flip at the end of the loop
            elif self.current_screen == "preview_stl":
                self.draw_stl_preview_screen()

            # Centralized rendering step
            self.render_and_flip_screen()
            
            self.clock.tick(30)

        # --- Cleanup on Exit ---
        print("Cleaning up and exiting...")
        neo.fill_strip(0, 0, 0)
        neo.update_strip()
        
        if self.cap0.isOpened():
            self.cap0.release()
        if self.cap2.isOpened():
            self.cap2.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    FullscreenUI()