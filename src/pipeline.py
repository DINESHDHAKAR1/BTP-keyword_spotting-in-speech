import yaml
import torch
from torch.utils.data import DataLoader
import logging
import os
from .data_ingestion import DataIngestion
from .data_preprocessing import AudioDataset
from .training import ModelTrainer
from .model import DiffusionClassifier, SimpleScheduler
from .model_evaluation import ModelEvaluator

class KeywordSpottingPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger("KeywordSpottingPipeline")
        log_dir = self.config["general"]["log_dir"]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "pipeline.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.device = self.config["general"]["device"]
        self.data_ingestion = DataIngestion(
            dataset_path=self.config["data_ingestion"]["dataset_path"],
            test_size=self.config["data_ingestion"]["test_size"],
            random_state=self.config["data_ingestion"]["random_state"],
            log_dir=log_dir
        )

    def run(self):
        try:
            # Load and split data
            train_paths, test_paths, train_labels, test_labels, label_to_idx = self.data_ingestion.load_data()
            num_classes = len(label_to_idx)
            self.logger.info(f"Number of classes: {num_classes}")

            # Create datasets
            train_dataset = AudioDataset(
                file_paths=train_paths,
                labels=train_labels,
                label_to_idx=label_to_idx,
                sr=self.config["data_preprocessing"]["sampling_rate"],
                n_mels=self.config["data_preprocessing"]["n_mels"],
                n_fft=self.config["data_preprocessing"]["n_fft"],
                hop_length=self.config["data_preprocessing"]["hop_length"],
                spectrogram_shape=tuple(self.config["data_preprocessing"]["spectrogram_shape"]),
                log_dir=self.config["general"]["log_dir"]
            )
            test_dataset = AudioDataset(
                file_paths=test_paths,
                labels=test_labels,
                label_to_idx=label_to_idx,
                sr=self.config["data_preprocessing"]["sampling_rate"],
                n_mels=self.config["data_preprocessing"]["n_mels"],
                n_fft=self.config["data_preprocessing"]["n_fft"],
                hop_length=self.config["data_preprocessing"]["hop_length"],
                spectrogram_shape=tuple(self.config["data_preprocessing"]["spectrogram_shape"]),
                log_dir=self.config["general"]["log_dir"]
            )

            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.config["model_training"]["batch_size"], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config["model_evaluation"]["batch_size"], shuffle=False)

            # Initialize model trainer
            model_trainer = ModelTrainer(
                num_classes=num_classes,
                device=self.device,
                base_channels=self.config["model_architecture"]["unet_channels"],
                classifier_hidden_dim=self.config["model_architecture"]["classifier_hidden_dim"],
                dropout_rate=self.config["model_architecture"]["dropout_rate"],
                timesteps=self.config["model_architecture"]["timesteps"],
                beta_start=self.config["model_architecture"]["beta_start"],
                beta_end=self.config["model_architecture"]["beta_end"],
                learning_rate=self.config["model_training"]["learning_rate"],
                log_dir=self.config["general"]["log_dir"]
            )

            # Train the model
            train_losses, test_losses, train_accuracies, test_accuracies = model_trainer.train(
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=self.config["model_training"]["num_epochs"],
                lambda_cls=self.config["model_training"]["lambda_cls"],
                lambda_denoise=self.config["model_training"]["lambda_denoise"],
                model_path=self.config["model_training"]["model_path"]
            )

            # Evaluate the model
            model_evaluator = ModelEvaluator(
                model=model_trainer.model,
                scheduler=model_trainer.scheduler,
                device=self.device,
                log_dir=self.config["general"]["log_dir"]
            )
            evaluation_report = model_evaluator.evaluate(test_loader)

            # Plot training curves
            model_evaluator.plot_curves(train_losses, test_losses, train_accuracies, test_accuracies)

            self.logger.info("Pipeline execution completed")
            return evaluation_report

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def predict(self, audio_paths):
        try:
            # Create a temporary dataset for prediction
            label_to_idx = {label: idx for idx, label in enumerate(range(len(audio_paths)))}  # Dummy labels
            dataset = AudioDataset(
                file_paths=audio_paths,
                labels=[0] * len(audio_paths),  # Dummy labels
                label_to_idx=label_to_idx,
                sr=self.config["data_preprocessing"]["sampling_rate"],
                n_mels=self.config["data_preprocessing"]["n_mels"],
                n_fft=self.config["data_preprocessing"]["n_fft"],
                hop_length=self.config["data_preprocessing"]["hop_length"],
                spectrogram_shape=tuple(self.config["data_preprocessing"]["spectrogram_shape"]),
                log_dir=self.config["general"]["log_dir"]
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            # Load the trained model
            model_trainer = ModelTrainer(
                num_classes=len(label_to_idx),
                device=self.device,
                base_channels=self.config["model_architecture"]["unet_channels"],
                classifier_hidden_dim=self.config["model_architecture"]["classifier_hidden_dim"],
                dropout_rate=self.config["model_architecture"]["dropout_rate"],
                timesteps=self.config["model_architecture"]["timesteps"],
                beta_start=self.config["model_architecture"]["beta_start"],
                beta_end=self.config["model_architecture"]["beta_end"],
                learning_rate=self.config["model_training"]["learning_rate"],
                log_dir=self.config["general"]["log_dir"]
            )
            model_trainer.model.load_state_dict(torch.load(self.config["model_training"]["model_path"]))
            model_trainer.model.eval()

            predictions = []
            with torch.no_grad():
                for spectrograms, _ in loader:
                    spectrograms = spectrograms.to(self.device)
                    batch_size = spectrograms.size(0)
                    t = torch.randint(0, model_trainer.scheduler.num_timesteps, (batch_size,)).to(self.device)
                    noise = torch.randn_like(spectrograms).to(self.device)
                    noisy_spectrograms = model_trainer.scheduler.add_noise(spectrograms, noise, t)
                    logits, _ = model_trainer.model(noisy_spectrograms, t)
                    _, predicted = torch.max(logits, 1)
                    predictions.append(predicted.item())

            self.logger.info(f"Predictions made for {len(audio_paths)} audio files")
            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise