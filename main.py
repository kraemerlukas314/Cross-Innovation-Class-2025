import pygame
import os
import sys
import cv2
import numpy as np

# --- Dependency Check for numpy-stl ---
try:
    from stl import mesh
except ImportError:
    print("Error: Could not import 'mesh' from 'stl'. Ensure 'numpy-stl' is installed correctly.")
    print("Run 'pip install numpy-stl' in your terminal.")
    sys.exit(1)

# --- Dependency Check for scikit-image ---
try:
    from skimage import measure
except ImportError:
    print("Error: Could not import 'measure' from 'skimage'. This is needed for 3D model generation.")
    print("Run 'pip install scikit-image' in your terminal.")
    sys.exit(1)

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
        self.normalized_mask0 = None
        self.normalized_mask2 = None

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
        self.images['pause'] = self.load_image("pause.png")

        self.scan_rect = self.images['scan'].get_rect(topleft=(510, 100))
        self.premodel_rect = self.images['premodel'].get_rect(topleft=(165, 100))

        screen_width, screen_height = self.screen.get_size()
        self.cancel_rect = self.images['cancel'].get_rect(center=(screen_width // 2, screen_height // 2))
        self.pause_rect = self.images['pause'].get_rect(midbottom=(screen_width // 2, screen_height - 50))

    def draw_main_menu(self):
        """Draws the main menu screen."""
        self.screen.blit(self.images['main'], (0, 0))
        self.screen.blit(self.images['scan'], self.scan_rect.topleft)
        self.screen.blit(self.images['premodel'], self.premodel_rect.topleft)

    def draw_white_screen_with_cancel(self):
        """Draws a white screen with a cancel button."""
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.images['cancel'], self.cancel_rect.topleft)

    def draw_camera_preview(self):
        """Draws the live camera feeds."""
        self.screen.fill((255, 255, 255))
        if self.frame0 is not None:
            surf0 = self.cv_to_pygame(self.frame0)
            self.screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.frame2 is not None:
            surf2 = self.cv_to_pygame(self.frame2)
            self.screen.blit(surf2, surf2.get_rect(topright=(self.screen.get_width() - 50, 100)))
        self.screen.blit(self.images['pause'], self.pause_rect.topleft)

    def draw_mask_preview(self):
        """Draws the generated binary masks."""
        self.screen.fill((255, 255, 255))
        if self.mask0 is not None:
            surf0 = self.gray_to_pygame(self.mask0)
            self.screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.mask2 is not None:
            surf2 = self.gray_to_pygame(self.mask2)
            self.screen.blit(surf2, surf2.get_rect(topright=(self.screen.get_width() - 50, 100)))
        self.screen.blit(self.images['pause'], self.pause_rect.topleft)

    def draw_normalized_preview(self):
        """Draws the final, normalized masks before 3D generation."""
        self.screen.fill((255, 255, 255))
        if self.normalized_mask0 is not None:
            surf0 = self.gray_to_pygame(self.normalized_mask0)
            self.screen.blit(surf0, surf0.get_rect(topleft=(50, 100)))
        if self.normalized_mask2 is not None:
            surf2 = self.gray_to_pygame(self.normalized_mask2)
            self.screen.blit(surf2, surf2.get_rect(topright=(self.screen.get_width() - 50, 100)))
        self.screen.blit(self.images['pause'], self.pause_rect.topleft)

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

    def handle_events(self):
        """Handles all user input events."""
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

                elif self.current_screen == "preview_mask":
                    if self.pause_rect.collidepoint(event.pos):
                        self.normalized_mask0 = self.normalize_mask(self.mask0)
                        self.normalized_mask2 = self.normalize_mask(self.mask2)
                        self.current_screen = "normalized_mask"

                elif self.current_screen == "normalized_mask":
                    if self.pause_rect.collidepoint(event.pos):
                        self.generate_stl()
                        self.current_screen = "main_menu"

    def start_camera_feeds(self):
        """Initializes and opens camera devices."""
        self.cap0 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(2)
        if not self.cap0.isOpened():
            print("Failed to open /dev/video0")
        if not self.cap2.isOpened():
            print("Failed to open /dev/video2")

        self.camera_active = True
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
            self.camera_active = False
            self.cap0.release()
            self.cap2.release()
            cv2.destroyAllWindows()

    def process_mask(self, image):
        """Converts an image to a clean, filled, binary mask of the largest object."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

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

    def generate_stl(self):
        """
        Generates a 3D model (STL) from the side and top view masks using voxel carving.
        This method computes the visual hull from the two silhouettes.
        """
        if self.normalized_mask0 is None or self.normalized_mask2 is None:
            print("No normalized masks available to generate STL.")
            return

        print("Generating 3D model... This may take a moment.")

        # Define the resolution of our voxel space. Higher values are more detailed but slower.
        z_dim, y_dim, x_dim = 100, 100, 75

        # Resize masks to the voxel space dimensions for processing
        # cv2.resize takes (width, height)
        side_mask_resized = cv2.resize(self.normalized_mask0, (x_dim, y_dim), interpolation=cv2.INTER_NEAREST)
        top_mask_resized = cv2.resize(self.normalized_mask2, (x_dim, z_dim), interpolation=cv2.INTER_NEAREST)

        # Convert to boolean masks (True means solid)
        side_silhouette = side_mask_resized > 128  # Shape: (y_dim, x_dim)
        top_silhouette = top_mask_resized > 128    # Shape: (z_dim, x_dim)

        # Extrude side view (y,x) along the Z axis to form a 3D volume
        volume_from_side = np.broadcast_to(side_silhouette[np.newaxis, :, :], (z_dim, y_dim, x_dim))

        # Extrude top view (z,x) along the Y axis to form another 3D volume
        volume_from_top = np.broadcast_to(top_silhouette[:, np.newaxis, :], (z_dim, y_dim, x_dim))

        # The final object is the logical intersection (AND) of these two volumes
        volume = np.logical_and(volume_from_side, volume_from_top)

        # Pad the volume to ensure marching_cubes generates a closed mesh at the borders
        volume = np.pad(volume, 1, constant_values=False)

        # Use the marching cubes algorithm to extract a surface mesh from the voxel volume
        # The `spacing` argument scales the model to reflect the original aspect ratio
        spacing_z = 400.0 / z_dim
        spacing_y = 400.0 / y_dim
        spacing_x = 300.0 / x_dim
        verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, spacing=(spacing_z, spacing_y, spacing_x))

        # Create the STL mesh object
        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            stl_mesh.vectors[i] = verts[f]

        # Save the final mesh to an STL file
        stl_mesh.save('/home/pi/repo/model.stl')
        print("STL file saved as 'model.stl'")


    def update_camera_frames(self):
        """Reads the latest frames from the active cameras."""
        if self.camera_active:
            ret0, frame0 = self.cap0.read()
            ret2, frame2 = self.cap2.read()
            if ret0:
                self.frame0 = frame0.copy()
            if ret2:
                self.frame2 = frame2.copy()

    def main_loop(self):
        """The main application loop."""
        while self.running:
            self.handle_events()
            self.update_camera_frames()

            # --- Screen Drawing ---
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

            pygame.display.flip()
            self.clock.tick(30)

        # --- Cleanup on Exit ---
        if hasattr(self, 'cap0') and self.cap0.isOpened():
            self.cap0.release()
        if hasattr(self, 'cap2') and self.cap2.isOpened():
            self.cap2.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    FullscreenUI()
