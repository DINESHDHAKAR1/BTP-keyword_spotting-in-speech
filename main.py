from src import KeywordSpottingPipeline

if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = KeywordSpottingPipeline(config_path="config.yaml")
    evaluation_report = pipeline.run()

    # Print evaluation results
    print("Evaluation Report:")
    print(f"Accuracy: {evaluation_report['accuracy']:.2f}%")
    print("Confusion Matrix:\n", evaluation_report['confusion_matrix'])
    print("Classification Report:\n", evaluation_report['classification_report'])

    # Example prediction (optional)
    sample_audio = ["E:/speech_commands_v0.01/yes/0a2b400e_nohash_0.wav"]
    predictions = pipeline.predict(sample_audio)
    print("Sample Predictions:", predictions)