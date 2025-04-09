# model_utils.py

import os
import tensorflow as tf
import shutil

def save_model(model, path, overwrite=True, save_format='h5'):
    """
    Save a Keras model to disk, handling overwrites and format suffixes.

    Parameters:
        model (tf.keras.Model): Model to save
        path (str): File or directory path
        overwrite (bool): Whether to overwrite if exists
        save_format (str): 'h5' or 'tf'
    """
    if save_format not in ['h5', 'tf']:
        raise ValueError("save_format must be 'h5' or 'tf'")

    # Normalize path based on format
    if save_format == 'h5':
        if not path.endswith('.h5'):
            path += '.h5'
    elif save_format == 'tf':
        if path.endswith('.h5'):
            path = path[:-3]

    # Handle overwriting
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"'{path}' already exists and overwrite=False.")
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    # Save the model
    model.save(path, save_format=save_format)
    print(f"✅ Model saved to: {path}")
