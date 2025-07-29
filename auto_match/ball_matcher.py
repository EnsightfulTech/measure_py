from auto_match.cal_distance import find_circles
from auto_match.edge_drawing import CircleEdgeDrawDetector
import numpy as np
import cv2
from logging import info, error
from icecream import ic


def create_combined_bins_with_indices(left_points, right_points, tolerance=1.5):
    bins = {}
    for i, point in enumerate(np.vstack((left_points, right_points))):
        y_coord = point[1]
        is_left = i < len(left_points)
        index = i if is_left else i - len(left_points)
        assigned = False
        for bin_y in bins:
            if abs(bin_y - y_coord) <= tolerance:
                bins[bin_y].append((point, index, is_left))
                assigned = True
                break
        if not assigned:
            bins[y_coord] = [(point, index, is_left)]
    return bins

class CircleDetector:
    def __init__(self, left_image, right_image=None) -> None:
        """
        left_image: left RGB image
        right_image: right RGB image
        """
        self.left_image = left_image
        self.right_image = right_image

        self.left_radius = []
        self.left_points = []
        self.right_points = []

        self.raw_right_points = []

    def detect_left_circles(self):
        """
        Return left image detected result
        """
        # configration
        min_radius = 5  # 最小半径
        max_radius = 100  # 最大半径
        min_points = 50 # 最小轮廓点数50
        max_offset = 10  # 最大偏移量10
        thrsh = 220

        left_radius, left_points, save_imageL = find_circles(self.left_image, min_points, max_offset, min_radius, max_radius, thrsh)
        right_radius, right_points, save_imageR = find_circles(self.right_image, min_points, max_offset, min_radius, max_radius, thrsh)

        ic(left_points)
        ic(right_points)
        self.raw_right_points = right_points

        # match every pair of points
        for left_r, left_p in zip(left_radius, left_points):
            right_p = self.get_matched_point(left_p)
            # only return matched ball pairs
            if right_p is not None:
                self.left_radius.append(left_r)
                self.left_points.append(left_p)
                self.right_points.append(right_p)


        if len(self.left_points) == 0:
            error("No matched ball detected! Possible incorrect rectify.")
        return self.left_radius, self.left_points, self.right_points

    def detect_single_image(self):
        """
         Return single image detected result
       """
        # configration
        detector = CircleEdgeDrawDetector(img=self.left_image)
        left_radius, left_points, save_imageL = detector.detect_circle()
        ic(left_points)

        # match every pair of points
        for left_r, left_p in zip(left_radius, left_points):
            self.left_radius.append(left_r)
            self.left_points.append(left_p)


        if len(self.left_points) == 0:
            error("No matched ball detected! Possible incorrect rectify.")
        return self.left_radius, self.left_points

    def get_matched_point(self, left_point: np.ndarray) -> np.ndarray:
        """
        matched left to right detected circles 
        match by y_coord
        """ 
        for right_point in self.raw_right_points:
            if abs(left_point[1]-right_point[1]) < 3.0:
                # matched point
                return right_point
        return None

    def detect_edgedraw(self):
        detector = CircleEdgeDrawDetector(img=self.left_image)
        left_radius, left_points, save_imageL = detector.detect_circle()

        detector = CircleEdgeDrawDetector(img=self.right_image)
        right_radius, right_points, save_imageR = detector.detect_circle()

        ic(left_points)
        ic(right_points)
        self.raw_right_points = right_points


        bins = create_combined_bins_with_indices(left_points, right_points)

        for bin_points in bins.values():
            left_bin_points = [p for p in bin_points if p[2]]  # is_left is True
            right_bin_points = [p for p in bin_points if not p[2]]  # is_left is False

            left_bin_points.sort(key=lambda x: x[0][0])
            right_bin_points.sort(key=lambda x: x[0][0])

            matched_left_id = set()
            matched_right_id = set()

            for i in range(min(len(left_bin_points), len(right_bin_points))):
                left_point, left_index, _ = left_bin_points[i]
                right_point, right_index, _ = right_bin_points[i]

                if left_index not in matched_left_id and right_index not in matched_right_id:
                    self.left_radius.append(left_radius[left_index])
                    self.left_points.append(left_points[left_index])
                    self.right_points.append(right_points[right_index])
                    matched_left_id.add(left_index)
                    matched_right_id.add(right_index)


        if len(self.left_points) == 0:
            error("No matched ball detected! Possible incorrect rectify.")
        return self.left_radius, self.left_points, self.right_points


if __name__ == "__main__":
    left = r"C:\Users\xiahlXIAHaolin\DCIM\A_17011623721705616.jpg"
    right = r"C:\Users\xiahlXIAHaolin\DCIM\D_17011623721705616.jpg"
    left = cv2.imread(left)
    right = cv2.imread(right)


    circle_detector = CircleDetector(left, right)
    lp = circle_detector.detect_left_circles()

    for p in lp:
        print(p)
        res = circle_detector.get_matched_point(p)
        if res is not None:
            print(res)
        else:
            # raise error
            pass
