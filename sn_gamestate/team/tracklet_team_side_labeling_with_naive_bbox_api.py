import pandas as pd
import torch
import numpy as np
from tracklab.pipeline.videolevel_module import VideoLevelModule
import logging


log = logging.getLogger(__name__)


INPUT_IMAGE_WIDTH = 1280


def center_point(bbox: list[int]):
    # Input: bbox [left, top, width, height]
    # Output: center point of bbox [x, y]
    return {"x": bbox[0] + bbox[2] / 2, "y": bbox[1] + bbox[3] / 2}


class TrackletTeamSideLabelingWithNaiveBbox(VideoLevelModule):
    """
    bbox_detector (yolov8) で検出bboxに基づいて team_cluster 0,1 のどちらが left or right なのか決定する
    """

    input_columns = ["track_id", "team_cluster", "bbox_ltwh", "role"]
    output_columns = ["team"]

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

        if "track_id" not in detections.columns:
            return detections

        team_a = detections[detections.team_cluster == 0]
        team_b = detections[detections.team_cluster == 1]
        xa_coordinates = [
            center_point(bbox).get("x") if isinstance(bbox, dict) else np.nan
            for bbox in team_a.bbox_ltwh
        ]  # [left_top_x, left_top_y, width, height]
        xb_coordinates = [
            center_point(bbox).get("x") if isinstance(bbox, dict) else np.nan
            for bbox in team_b.bbox_ltwh
        ]  # (x, y) are the center of a bbox

        avg_a = np.nanmean(xa_coordinates)
        avg_b = np.nanmean(xb_coordinates)

        if avg_a > avg_b:
            detections.loc[team_a.index, "team"] = ["right"] * len(team_a)
            detections.loc[team_b.index, "team"] = ["left"] * len(team_b)
        else:
            detections.loc[team_a.index, "team"] = ["left"] * len(team_a)
            detections.loc[team_b.index, "team"] = ["right"] * len(team_b)

        # Goalkeeper labeling
        goalkeepers = detections[detections.role == "goalkeeper"].dropna(
            subset=["bbox_ltwh"]
        )
        gk_team = goalkeepers.bbox_ltwh.apply(
            lambda bbox: (
                "right"
                if (
                    center_point(bbox).get("x") > INPUT_IMAGE_WIDTH / 2
                )  # 画像の中心より右側が多めなら右チーム
                else "left"
            )
        )
        detections.loc[goalkeepers.index, "team"] = gk_team

        return detections
