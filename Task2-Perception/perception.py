import math
import cv2
import numpy as np
import sys


class RoboGambit_Perception:

    def __init__(self):
        # PARAMETERS - Camera intrinsics provided by organisers (DO NOT MODIFY)
        self.camera_matrix = np.array([
            [1030.4890823364258, 0, 960],
            [0, 1030.489103794098, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((1, 5))

        # INTERNAL VARIABLES
        self.corner_world = {
            21: (350, 350),
            22: (350, -350),
            23: (-350, -350),
            24: (-350, 350)
        }
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        self.H_matrix = None

        self.board = np.zeros((6, 6), dtype=int)

        # ARUCO DETECTOR
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        print("Perception Initialized")


    # DO NOT MODIFY THIS FUNCTION
    def prepare_image(self, image):
        """
        DO NOT MODIFY.
        Performs camera undistortion and grayscale conversion.
        """
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        gray_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        return undistorted_image, gray_image


    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates into world coordinates using homography.
        Steps:
        1. Ensure homography matrix has been computed.
        2. Format pixel point for cv2.perspectiveTransform().
        3. Return transformed world coordinates.
        """
        if self.H_matrix is None:
            print("Homography matrix not computed yet.")
            return None, None

        # Format as (1, 1, 2) array required by perspectiveTransform
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        # Apply homography to get world coordinates
        world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)

        world_x = world_point[0][0][0]
        world_y = world_point[0][0][1]

        return world_x, world_y


    # PARTICIPANTS MODIFY THIS FUNCTION
    def process_image(self, image):
        """
        Main perception pipeline.
        """

        self.board[:] = 0

        # Preprocess image (Do not modify)
        undistorted_image, gray_image = self.prepare_image(image)

        # --- Detect ArUco markers ---
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)

        if ids is None:
            print("No ArUco markers detected.")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        ids_flat = ids.flatten()

        # --- Extract corner marker pixels (IDs 21-24) ---
        # Each corner marker center is the mean of its 4 detected corners
        self.corner_pixels = {}
        for i, marker_id in enumerate(ids_flat):
            if marker_id in self.corner_world:
                marker_corners = corners[i][0]       # shape (4, 2)
                cx = float(np.mean(marker_corners[:, 0]))
                cy = float(np.mean(marker_corners[:, 1]))
                self.corner_pixels[marker_id] = (cx, cy)
                print(f"Corner marker {marker_id} detected at pixel ({cx:.1f}, {cy:.1f})")

        # --- Build pixel and world matrices ---
        self.pixel_matrix = []
        self.world_matrix = []
        for marker_id, pixel_coord in self.corner_pixels.items():
            self.pixel_matrix.append(list(pixel_coord))
            self.world_matrix.append(list(self.corner_world[marker_id]))

        # --- Compute homography matrix ---
        # Requires all 4 corner markers for a reliable mapping
        if len(self.pixel_matrix) >= 4:
            pixel_pts = np.array(self.pixel_matrix, dtype=np.float32)
            world_pts = np.array(self.world_matrix, dtype=np.float32)
            # Maps pixel coordinates → world coordinates (mm)
            self.H_matrix, mask = cv2.findHomography(pixel_pts, world_pts)
            print(f"Homography computed using {len(self.pixel_matrix)} corner markers.")
        else:
            print(f"WARNING: Only {len(self.pixel_matrix)}/4 corner markers found. Homography not computed.")

        # --- Convert piece markers (IDs 1-10) to world coordinates ---
        if self.H_matrix is not None:
            for i, marker_id in enumerate(ids_flat):
                if 1 <= marker_id <= 10:
                    marker_corners = corners[i][0]   # shape (4, 2)
                    cx = float(np.mean(marker_corners[:, 0]))
                    cy = float(np.mean(marker_corners[:, 1]))

                    world_x, world_y = self.pixel_to_world(cx, cy)

                    if world_x is not None:
                        self.place_piece_on_board(int(marker_id), world_x, world_y)
                        # Draw world coords on image for debugging
                        label = f"ID:{marker_id} ({world_x:.0f},{world_y:.0f})"
                        cv2.putText(undistorted_image, label,
                                    (int(cx) - 50, int(cy) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print final board state
        print("\n--- Board State (row 0 = Rank 6, col 0 = File A) ---")
        print(self.board)

        # Visualization (Do not modify)
        res = cv2.resize(undistorted_image, (1152, 648))
        cv2.imshow("Detected Markers", res)
        self.visualize_board()


    def place_piece_on_board(self, piece_id, x_coord, y_coord):
        """
        Places detected piece on the closest board square.

        Board definition:
          - 6x6 grid, square size = 100mm
          - Board spans x: -300 to +300,  y: -300 to +300 (world mm)
          - File A (col 0) is leftmost  (x ~ -250), File F (col 5) is rightmost (x ~ +250)
          - Rank 1 (row 5) is bottom    (y ~ -250), Rank 6 (row 0) is top       (y ~ +250)
          - Array element [0][0] = A6 (top-left from white's side)
          - Array element [5][0] = A1 (bottom-left, white home)
        """
        board_origin = -300.0   # world coordinate of board edge
        square_size  = 100.0

        # Column (file A=0 → F=5): increases with x
        col = int((x_coord - board_origin) / square_size)

        # Row (rank 6=0 → rank 1=5): row 0 is top (largest y)
        row = int((300.0 - y_coord) / square_size)

        # Clamp to valid [0, 5] range
        col = max(0, min(5, col))
        row = max(0, min(5, row))

        self.board[row][col] = piece_id
        print(f"  Piece {piece_id:2d} → board[{row}][{col}]  (world: x={x_coord:.1f}, y={y_coord:.1f})")


    # DO NOT MODIFY THIS FUNCTION
    def visualize_board(self):
        """
        Draw a simple 6x6 board with detected piece IDs
        """
        cell_size = 80
        board_img = np.ones((6 * cell_size, 6 * cell_size, 3), dtype=np.uint8) * 255

        for r in range(6):
            for c in range(6):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cv2.rectangle(board_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

                piece = int(self.board[r][c])
                if piece != 0:
                    cv2.putText(board_img, str(piece), (x1 + 25, y1 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Game Board", board_img)


# DO NOT MODIFY
def main():
    # To run code, use python/python3 perception.py path/to/image.png
    if len(sys.argv) < 2:
        print("Usage: python perception.py image.png")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    perception = RoboGambit_Perception()
    perception.process_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
