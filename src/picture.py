import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Picture:
    
    def __init__(self, base_directory) -> None:
        self.base_directory = base_directory
        self.pictures = []

    def get_directory_names(self) -> list[str]:
        """
        Used to get directory names inside the base directory.
        """
        all_items = os.listdir(self.base_directory)
        return [item for item in all_items if os.path.isdir(os.path.join(self.base_directory, item))]

    def get_file_names(self, directory_name: str) -> list[str]:
        """
        Used to get file names inside a directory which is in the base directory.
        """
        directory_path = os.path.join(self.base_directory, directory_name)
        all_items = os.listdir(directory_path)
        return [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]
    
    def set_pictures(self, pictures: list) -> None:
        """
        Used to set pictures.
        """
        self.pictures = pictures

    def load_pictures(self, directory_name: str, extension: str = ".BMP") -> None:
        """
        Used to load pictures inside a directory which is in the base directory and store them in a list.
        """
        directory_path = os.path.join(self.base_directory, directory_name)
        picture_files = self.get_file_names(directory_name)
        pictures = []

        for file in picture_files:
            if file.endswith(extension):
                file_path = os.path.join(directory_path, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
                pictures.append({
                    "name": file[:-4],
                    "image": image
                })
        
        self.set_pictures(pictures)

    def average_picture(self) -> np.ndarray:
        """
        Return the average of the loaded pictures.
        """
        if not self.pictures:
            raise ValueError("No pictures loaded.")

        # Calculate the average of the loaded pictures
        pictures = [picture["image"] for picture in self.pictures]
        average = np.mean(pictures, axis=0).astype(np.uint8)
        return average
    
    def display_picture(self, picture: np.ndarray) -> None:
        """
        Display the picture in a window.
        """
        cv2.imshow("Picture", picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_picture(self, picture: np.ndarray, directory_name: str, prefix_name: str) -> None:
        """
        Save the picture in a folder in the base directory with a prefix name.
        """
        directory_path = os.path.join(self.base_directory, directory_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        file_path = os.path.join(directory_path, prefix_name + ".png")
        cv2.imwrite(file_path, picture)
    
    def traitement(self, treshold_min: int, treshold_max: int, smooth: int) -> None:
        """
        For every picture in the list, find 2 areas with high intensity and determine their equations f(x)=ax+b.
        """
        pictures = self.pictures
        
        """
        fig, axs = plt.subplots(2, len(pictures), figsize=(15, 10))
        # make them all share the same x axis
        for ax in axs.flat:
            ax.set(xlabel='x', ylabel='intensity')
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        """
        fullscreen_size = (16, 9)
        fig = plt.figure(figsize=fullscreen_size)
        gs = fig.add_gridspec(2, len(pictures), hspace=0, wspace=0)
        axs = gs.subplots(sharex='col', sharey='row')
        for ax in axs.flat:
            ax.set(xlabel='x (px)', ylabel='intensity (from 0 to 255)')
        for ax in axs.flat:
            ax.label_outer()
        

        for picture_number, picture in enumerate(pictures):
            
            print(f"\nProcessing picture {picture['name']} :")

            image = picture["image"]
            # Threshold the image (convert to binary)
            _, thresholded_image = cv2.threshold(image, treshold_min, treshold_max, cv2.THRESH_BINARY)

            # Find contours of the white regions
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            angle = [100,100]
            # Sort contours by area
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for i in range(min(2, len(sorted_contours))):
                contour = sorted_contours[i]

                # Fit a line to the contour points using least squares
                rows, cols = image.shape
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / vx
                intercept = y - slope * x

                # Now 'slope' and 'intercept' contain the coefficients 'a' and 'b' of the line equation f(x) = ax + b
                print(f"Equation for area {i + 1}: f(x) = {slope}x + {intercept}")
                
                # Compute the len of the line with the criteria end or start when the derivative is max and min
                # Get values on the line
                x_values, y_values, intensity = [], [], []
                for x in range(cols):
                    x_values.append(x)
                    y_values.append(int(slope * x + intercept))
                    intensity.append(image[y_values[-1], x_values[-1]])
                    if y < 0 or y >= rows:
                        continue
                
                # Make intensity values smoother with the same shape
                # Define a smoothing window (e.g., moving average)
                window = np.ones(smooth) / smooth
                # Apply the moving average filter to smooth the signal
                intensity = np.convolve(intensity, window, mode='same')
                print(f"Intensity values: {intensity}")
                
                # Sublot the intensity
                axs[i, picture_number].plot(x_values, intensity)
                if i == 0:
                    axs[i, picture_number].set_title(f"Areas for {picture['name']}")
                #elif i == 1:
                #    axs[i, picture_number].set_title(f"Area {i + 1} with {picture['name']}")
                # Find the derivative of the line, need to increase the derivative order to find the max and min
                derivative = []
                distance = 1
                for _j in range(distance, len(x_values) - distance):
                    derivative_value = (intensity[_j + distance] - intensity[_j - distance])
                    derivative.append(derivative_value)
                derivative = np.array(derivative)
                # Plot the derivative on another axis
                axs[i, picture_number].plot(x_values[:-distance*2], derivative)
                # Find the max and min of the derivative
                max_derivative = np.max(derivative)
                min_derivative = np.min(derivative)
                # Find the index of the max and min of the derivative
                max_derivative_index = np.where(derivative == max_derivative)[0][0]
                min_derivative_index = np.where(derivative == min_derivative)[0][0]
                # Find the point of the line with the max and min of the derivative
                point_max_derivative = (x_values[max_derivative_index], y_values[max_derivative_index])
                point_min_derivative = (x_values[min_derivative_index], y_values[min_derivative_index])
                # Draw the point on the picture
                cv2.circle(image, point_max_derivative, 10, (255, 255, 255), -1)
                cv2.circle(image, point_min_derivative, 10, (255, 255, 255), -1)
                # Compute the len of the line
                len_line = np.sqrt((point_max_derivative[0] - point_min_derivative[0]) ** 2 + (point_max_derivative[1] - point_min_derivative[1]) ** 2)
                print(f"Len of the line: {len_line}")
                
                # Draw the line on the picture
                point1 = (0, int(intercept))
                point2 = (cols - 1, int(slope * (cols - 1) + intercept))
                cv2.line(image, point1, point2, (255, 0, 0), 2)
                
                # Compute the angle of the line
                angle[i] = np.arctan(slope) * 180 / np.pi
            
            # Determine the angle between the 2 lines
            result_angle = np.abs(angle[0] - angle[1])
            print(f"Angle between the 2 lines: {result_angle} degrees")
            
            name = "post_traitement"
            self.save_picture(image, name, picture["name"])
            
        # Render the plot picture as a PNG file
        directory_path = self.base_directory + "/post_traitement/intensity_len_comparison.png"
        plt.savefig(directory_path)