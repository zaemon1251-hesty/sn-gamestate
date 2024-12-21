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
        frame = cv2.imread(image_path)
        if frame is None:
            log.warning(f"Image at {image_path} could not be loaded.")
            return np.array([0, 0, 0])

        l, t, r, b = bbox.ltrb(
            image_shape=(frame.shape[1], frame.shape[0]), rounded=True
        )
        cropped_img = frame[t:b, l:r]

        if cropped_img.size == 0:
            return np.array([0, 0, 0])

        avg_color = np.mean(cropped_img, axis=(0, 1))
        return avg_color

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if detections.empty:
            raise ValueError("No detections found in the DataFrame.")

        if metadatas.empty:
            raise ValueError("No metadata found in the DataFrame.")

        player_detections = detections  # [detections.role == "player"]
        if player_detections.empty:
            print(detections.role.count())
            raise ValueError("No player detections found in the DataFrame.")

        color_list = []
        for idx, row in player_detections.iterrows():
            # Use frame (stringified) to match image_id in metadata DataFrame
            metadata_row = metadatas[metadatas.id == row.image_id]
            if metadata_row.empty:
                print(f"Metadata not found for frame: {row.image_id}, {metadatas.id}")
                continue

            image_path = metadata_row["file_path"].values[0]
            bbox = row.bbox
            avg_color = self.get_average_color(image_path, bbox)
            color_list.append({"track_id": row.track_id, "avg_color": avg_color})

        if not color_list:
            detections["team_cluster"] = np.nan
            print("Warnings: all of metadata_row are empty.!!!!!")
            return detections

        color_tracklet = (
            pd.DataFrame(color_list).groupby("track_id").mean().reset_index()
        )

        if len(color_tracklet) == 1:
            color_tracklet["team_cluster"] = 0
        else:
            colors = np.vstack(color_tracklet.avg_color.values)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(colors)
            color_tracklet["team_cluster"] = kmeans.labels_

        detections.drop(columns=["team_cluster"], errors="ignore", inplace=True)
        detections = detections.merge(
            color_tracklet[["track_id", "team_cluster"]],
            on="track_id",
            how="left",
            sort=False,
        )

        return detections
