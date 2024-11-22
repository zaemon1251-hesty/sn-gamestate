import logging
from typing import Generator, Iterable, List, TypeVar
import pandas as pd
import cv2
import numpy as np
import supervision as sv
import torch
import umap.umap_ as umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from tracklab.pipeline.videolevel_module import VideoLevelModule

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        """
        Initialize the TeamClassifier with device and batch size.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
            batch_size (int): The batch size for processing images.
        """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(
            device
        )
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


og = logging.getLogger(__name__)


class TeamSiglipUmapKmeans(VideoLevelModule):
    """
    Inspired by https://github.com/roboflow/sports
    """

    input_columns = ["track_id", "bbox_ltwh", "role", "image_id"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cuda"

        team_classifier = TeamClassifier(device=self.device)
        self.team_classifier = team_classifier
        super().__init__()

    def get_crop(self, image_path, bbox):
        """
        Extract the average color from a given bounding box (bbox) in the frame.

        :param image_path: Path to the current image file.
        :param bbox: Bounding box coordinates [x_min, y_min, width, height].
        :return: Average BGR color as a numpy array [B, G, R].
        """
        # Load the image from the file path
        frame = cv2.imread(image_path)
        if frame is None:
            logging.warning(f"Image at {image_path} could not be loaded.")
            return np.array([0, 0, 0])  # Return black if image couldn't be loaded

        l, t, r, b = bbox.ltrb(
            image_shape=(frame.shape[1], frame.shape[0]), rounded=True
        )
        cropped_img = frame[t:b, l:r]

        if cropped_img.size == 0:
            return np.array([0, 0, 0])  # Return black if bbox is invalid

        return cropped_img

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Filter player detections
        player_detections = detections[detections.role == "player"]

        crop_list = []

        for idx, row in player_detections.iterrows():
            metadata_row = metadatas[metadatas["id"] == row.image_id]
            if metadata_row.empty:
                logging.warning(f"Metadata not found for frame: {row.image_id}")
                continue

            image_path = metadata_row["file_path"].values[0]
            bbox = row.bbox
            crop = self.get_crop(image_path, bbox)
            crop_list.append({"track_id": row.track_id, "crop": crop})

        if not crop_list:
            detections["team_cluster"] = np.nan
            return detections

        # 平均プーリングはうまく動かないことがわかったので、track_idごとに代表的なcropを取得
        color_tracklet = (
            pd.DataFrame(crop_list).groupby("track_id").first().reset_index()
        )

        if len(color_tracklet) == 1:
            color_tracklet["team_cluster"] = 0
        else:
            crops = color_tracklet.crop.values
            crops = [crop for crop in crops]
            self.team_classifier.fit(crops)
            color_tracklet["team_cluster"] = self.team_classifier.predict(crops)

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(
            color_tracklet[["track_id", "team_cluster"]],
            on="track_id",
            how="left",
            sort=False,
        )

        return detections
