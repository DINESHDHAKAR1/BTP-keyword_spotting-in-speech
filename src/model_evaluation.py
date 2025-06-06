import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os
from .model import SimpleScheduler

class ModelEvaluator:
    def __init__(self, model, scheduler, device="cuda", log_dir="logs"):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.logger = logging.getLogger("ModelEvaluator")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "model_evaluation.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def evaluate(self, test_loader, batch_size=128):
        try:
            self.model.eval()
            all_preds = []
            all_labels = []
            total_loss = 0
            criterion = torch.nn.CrossEntropyLoss()

            with torch.no_grad():
                for spectrograms, labels in test_loader:
                    spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                    batch_size = spectrograms.size(0)
                    t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,)).to(self.device)
                    noise = torch.randn_like(spectrograms).to(self.device)
                    noisy_spectrograms = self.scheduler.add_noise(spectrograms, noise, t)
                    logits, _ = self.model(noisy_spectrograms, t)

                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Compute metrics
            accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
            cm = confusion_matrix(all_labels, all_preds)
            report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(len(set(all_labels)))], zero_division=0)

            self.logger.info(f"Evaluation Results: Accuracy: {accuracy:.2f}%")
            self.logger.info("Confusion Matrix:\n" + str(cm))
            self.logger.info("Classification Report:\n" + report)

            return {
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "classification_report": report,
                "loss": total_loss / len(test_loader)
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise

    @staticmethod
    def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies):
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [tl[0] for tl in train_losses], label="Train Classification Loss")
        plt.plot(epochs, [tl[1] for tl in train_losses], label="Train Denoising Loss")
        plt.plot(epochs, [tl[0] for tl in test_losses], label="Test Classification Loss")
        plt.plot(epochs, [tl[1] for tl in test_losses], label="Test Denoising Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy")
        plt.plot(epochs, test_accuracies, label="Test Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.show()