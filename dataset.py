import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Callable
from PIL import Image

class VisualOdometryDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        transform: Callable,
        sequence_length: int,
        validation: bool = False
    ) -> None:
        self.sequences = []

        directories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        for subdir in directories:
            aux_path = f"{dataset_path}/{subdir}"

            # read data
            rgb_paths = self.read_images_paths(aux_path)

            if not validation:
                ground_truth_data = self.read_ground_truth(aux_path)
                interpolated_ground_truth = self.interpolate_ground_truth(
                    rgb_paths, ground_truth_data
                )

                # Crear secuencias
                for i in range(0, len(rgb_paths) - sequence_length + 1):
                    sequence = rgb_paths[i:i + sequence_length]
                    ground_truth_seq = interpolated_ground_truth[i:i + sequence_length]
                    self.sequences.append((sequence, ground_truth_seq))

        self.transform = transform
        self.sequence_length = sequence_length
        self.validation = validation

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        sequence_images, ground_truth_seq = self.sequences[idx]

        images = []
        for _, image_path in sequence_images:
            image = Image.open(image_path)  # Cargar imagen usando PIL
            image = self.transform(image)  # Aplicar transformaciones
            images.append(image)

        images = torch.stack(images)
        
        # Extraer solo las listas de números reales de ground_truth_seq
        ground_truth_values = [pos for _, pos in ground_truth_seq]

        # Convertir ground_truth_values en un tensor 2D
        try:
            ground_truth_pos = torch.tensor(ground_truth_values, dtype=torch.float32)
        except Exception as e:
            print(f"Error al convertir ground_truth_values en tensor: {e}")
            raise e

        timestamp = sequence_images[-1][0]

        return images, ground_truth_pos, timestamp

    def read_images_paths(self, dataset_path: str) -> List[Tuple[float, str]]:
        paths = []

        with open(f"{dataset_path}/rgb.txt", "r") as file:
            for line in file:
                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                image_path = f"{dataset_path}/{line[1]}"

                paths.append((timestamp, image_path))

        return paths

    def read_ground_truth(self, dataset_path: str) -> List[Tuple[float, List[float]]]:
        ground_truth_data = []

        with open(f"{dataset_path}/groundtruth.txt", "r") as file:
            for line in file:
                if line.startswith("#"):  # Skip comment lines
                    continue

                line = line.strip().split()
                timestamp = float(line[0])
                position = list(map(float, line[1:]))
                ground_truth_data.append((timestamp, position))

        return ground_truth_data

    def interpolate_ground_truth(
            self,
            rgb_paths: List[Tuple[float, str]],
            ground_truth_data: List[Tuple[float, List[float]]]
    ) -> List[Tuple[float, List[float]]]:
        rgb_timestamps = [rgb_path[0] for rgb_path in rgb_paths]
        ground_truth_timestamps = [item[0] for item in ground_truth_data]

        # Interpolate ground truth positions for each RGB image timestamp
        interpolated_ground_truth = []

        for rgb_timestamp in rgb_timestamps:
            nearest_idx = np.argmin(
                np.abs(np.array(ground_truth_timestamps) - rgb_timestamp)
            )

            interpolated_position = ground_truth_data[nearest_idx]
            interpolated_ground_truth.append(interpolated_position)

        return interpolated_ground_truth
