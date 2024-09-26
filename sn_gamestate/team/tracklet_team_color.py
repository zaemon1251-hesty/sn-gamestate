import pandas as pd
import numpy as np
import cv2  # OpenCV for image processing
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
from sklearn.cluster import KMeans  # for clustering

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class TrackletTeamClusteringByColor(VideoLevelModule):
    """
    This module performs KMeans clustering on the average color within bounding boxes (bboxes)
    to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformed into 'left' and 'right' in a separate module.
    """

    input_columns = ["track_id", "bbox", "frame", "role"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        super().__init__()

    def get_average_color(self, frame, bbox):
        """
        Extract the average color from a given bounding box (bbox) in the frame.

        :param frame: The current video frame.
        :param bbox: Bounding box coordinates [x_min, y_min, width, height].
        :return: Average BGR color as a numpy array [B, G, R].
        """
        x_min, y_min, width, height = map(int, bbox)
        cropped_img = frame[y_min : y_min + height, x_min : x_min + width]
        if cropped_img.size == 0:
            return np.array([0, 0, 0])  # Return black if bbox is invalid
        avg_color = np.mean(cropped_img, axis=(0, 1))  # Average over width and height
        return avg_color

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Filter player detections
        player_detections = detections[detections.role == "player"]

        # List to store the average color for each track_id
        color_list = []

        for idx, row in player_detections.iterrows():
            frame = metadatas[
                row.frame
            ]  # Assuming metadatas has frames as a list/dict of images
            bbox = row.bbox  # Bounding box for the player
            avg_color = self.get_average_color(frame, bbox)  # Compute average color
            color_list.append({"track_id": row.track_id, "avg_color": avg_color})

        if not color_list:  # Check if color_list is empty
            detections["team_cluster"] = (
                np.nan
            )  # Initialize 'team_cluster' with a default value
            return detections

        # Create a DataFrame with the average colors for each track_id
        color_tracklet = (
            pd.DataFrame(color_list).groupby("track_id").mean().reset_index()
        )

        if len(color_tracklet) == 1:  # Only one track_id and color
            color_tracklet["team_cluster"] = 0
        else:
            # Perform KMeans clustering on the average colors
            colors = np.vstack(color_tracklet.avg_color.values)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
            color_tracklet["team_cluster"] = kmeans.labels_

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(
            color_tracklet[["track_id", "team_cluster"]],
            on="track_id",
            how="left",
            sort=False,
        )

        return detections
