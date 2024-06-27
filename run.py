import os

import lightning as L

from model.hoax_detection_model import HoaxDetectionModel
from utils.preprocessor import Preprocessor

if __name__ == "__main__":
    
    dm = Preprocessor(
        batch_size = 10
    )
    
    model = HoaxDetectionModel(model_id = "indolem/indobert-base-uncased")
    
    trainer = L.Trainer(
        accelerator = 'gpu',
        max_epochs = 1,
        default_root_dir = 'logs/indobert'
    )
    
    # Ini bagian training 
    trainer.fit(model, datamodule = dm)

    # Testing model
    # trainer.test(datamodule = dm, ckpt_path = 'best')
    trainer.test(model=model, datamodule=dm, ckpt_path='best')
    