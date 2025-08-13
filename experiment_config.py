from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("C:/Users/prime/Desktop/luna25-baseline-public")
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("C:/Users/prime/Desktop/luna25-baseline-public/data/luna25_nodule_blocks")
        # self.DATADIR = Path("C:/Users/prime/Desktop/luna25-baseline-public/data/data_augmented")
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("C:/Users/prime/Desktop/luna25-baseline-public/csv")
        # self.CSV_DIR = Path("C:/Users/prime/Desktop/luna25-baseline-public/csv_augmented")        
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "valid.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "UNet3DWithQPF Original data 10 epochs"
        self.MODE = "UNet3DWithQPF" # 2D or 3D

        #For Resnet3d
        self.RESNET_DEPTH = 34

        #For QNN
        self.QNN_TARGET_IMAGE_FEATURES = 8  # How many features to get from the image (e.g., 2x2x2)
        self.METADATA_DIM = 15               # The number of features in your metadata
        self.QNN_N_LAYERS = 6               # Number of entangled layers in the quantum circuit
        self.N_QUBITS = 4    
        self.N_LAYERS = 4
        self.QNN_N_QUBITS = 5
        self.PQC_N_FEATURES = 4    # Number of features after PCA, must match n_qubits
        self.PQC_N_LAYERS = 3      # Number of layers in the PQC (start small, e.g., 3-5)

        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 10
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4


config = Configuration()
