import pandas as pd
import numpy as np
import cv2  # OpenCV for image processing
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
from sklearn.cluster import KMeans  # for clustering
import torch
import os

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class TrackletTeamClusteringByColor(VideoLevelModule):
    """
    This module performs KMeans clustering on the average color within bounding boxes (bboxes)
    to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformed into 'left' and 'right' in a separate module.
    """

    input_columns = ["track_id", "bbox_ltwh", "role", "image_id"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        super().__init__()

    def get_average_color(self, image_path, bbox):
        """
        Extract the average color from a given bounding box (bbox) in the frame.

        :param image_path: Path to the current image file.
        :param bbox: Bounding box coordinates [x_min, y_min, width, height].
        :return: Average BGR color as a numpy array [B, G, R].
        """
        # Load the image from the file path
        frame = cv2.imread(image_path)
        if frame is None:
            log.warning(f"Image at {image_path} could not be loaded.")
            return np.array([0, 0, 0])  # Return black if image couldn't be loaded

        l, t, r, b = bbox.ltrb(
            image_shape=(frame.shape[1], frame.shape[0]), rounded=True
        )
        cropped_img = frame[t:b, l:r]

        if cropped_img.size == 0:
            return np.array([0, 0, 0])  # Return black if bbox is invalid

        # Calculate the average color of the cropped image
        avg_color = np.mean(cropped_img, axis=(0, 1))  # Average over width and height
        return avg_color

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Filter player detections
        player_detections = detections[detections.role == "player"]

        # List to store the average color for each track_id
        color_list = []

        for idx, row in player_detections.iterrows():
            # Use frame (stringified) to match image_id in metadata DataFrame
            metadata_row = metadatas[metadatas["id"] == row.image_id]
            if metadata_row.empty:
                log.warning(f"Metadata not found for frame: {row.image_id}")
                continue

            image_path = metadata_row["file_path"].values[0]
            bbox = row.bbox  # Bounding box for the player
            avg_color = self.get_average_color(
                image_path, bbox
            )  # Compute average color
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
