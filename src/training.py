import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
from .model import DiffusionClassifier, SimpleScheduler

class ModelTrainer:
    def __init__(self, num_classes, device="cuda", base_channels=64, classifier_hidden_dim=128, dropout_rate=0.3,
                 timesteps=1000, beta_start=1e-4, beta_end=0.02, learning_rate=0.001, log_dir="logs"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DiffusionClassifier(
            num_classes=num_classes,
            base_channels=base_channels,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        self.scheduler = SimpleScheduler(timesteps, beta_start, beta_end, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger("ModelTrainer")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "model_training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger.info(f"Device in use {self.device}")

    def train(self, train_loader, test_loader, num_epochs, lambda_cls=1.0, lambda_denoise=1.0, model_path="model.pth"):
        try:
            train_losses = []
            test_losses = []
            train_accuracies = []
            test_accuracies = []

            for epoch in range(num_epochs):
                self.model.train()
                total_cls_loss = 0
                total_denoise_loss = 0
                correct = 0
                total = 0

                for spectrograms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()

                    # Add noise using the diffusion process
                    batch_size = spectrograms.size(0)
                    t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,)).to(self.device)
                    noise = torch.randn_like(spectrograms).to(self.device)
                    noisy_spectrograms = self.scheduler.add_noise(spectrograms, noise, t)

                    # Forward pass
                    logits, unet_out = self.model(noisy_spectrograms, t)

                    # Classification loss
                    cls_loss = self.criterion(logits, labels)
                    total_cls_loss += cls_loss.item()

                    # Denoising loss
                    denoise_loss = torch.mean((unet_out - spectrograms) ** 2)
                    total_denoise_loss += denoise_loss.item()

                    # Combined loss
                    loss = lambda_cls * cls_loss + lambda_denoise * denoise_loss
                    loss.backward()
                    self.optimizer.step()

                    # Compute accuracy
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                # Compute epoch metrics
                train_accuracy = 100 * correct / total
                train_losses.append((total_cls_loss / len(train_loader), total_denoise_loss / len(train_loader)))
                train_accuracies.append(train_accuracy)

                # Evaluate on test set
                test_accuracy = self.evaluate_epoch(test_loader)  # Remove the 't' parameter
                test_losses.append((total_cls_loss / len(train_loader), total_denoise_loss / len(train_loader)))
                test_accuracies.append(test_accuracy)

                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, Classification Loss: {total_cls_loss / len(train_loader):.4f}, "
                    f"Denoising Loss: {total_denoise_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%"
                )
                self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

            # Save the model
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Model saved to {model_path}")

            return train_losses, test_losses, train_accuracies, test_accuracies

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate_epoch(self, test_loader):  # Remove the 't' parameter
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for spectrograms, labels in test_loader:
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                batch_size = spectrograms.size(0)
                t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,)).to(self.device)  # Generate t for each batch
                noise = torch.randn_like(spectrograms).to(self.device)
                noisy_spectrograms = self.scheduler.add_noise(spectrograms, noise, t)
                logits, _ = self.model(noisy_spectrograms, t)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")