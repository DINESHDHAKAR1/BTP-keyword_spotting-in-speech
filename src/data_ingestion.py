import os
import logging
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, dataset_path, test_size=0.2, random_state=42, log_dir="logs"):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logging.getLogger("DataIngestion")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "data_ingestion.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def load_data(self):
        try:
            audio_data = []
            labels = []
            file_paths = []

            for class_name in os.listdir(self.dataset_path):
                class_path = os.path.join(self.dataset_path, class_name)
                if os.path.isdir(class_path):
                    for file_name in os.listdir(class_path):
                        if file_name.endswith(".wav"):
                            file_path = os.path.join(class_path, file_name)
                            file_paths.append(file_path)
                            labels.append(class_name)

            if not file_paths:
                raise ValueError("No .wav files found in the dataset path")

            # Create label-to-index mapping
            label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            self.logger.info(f"Loaded {len(file_paths)} audio files with {len(label_to_idx)} classes")

            # Split into train and test sets
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                file_paths, labels, test_size=self.test_size, stratify=labels, random_state=self.random_state
            )
            self.logger.info("Data split into training and testing sets")

            return train_paths, test_paths, train_labels, test_labels, label_to_idx

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise