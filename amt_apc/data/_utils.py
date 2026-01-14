import torch
import numpy as np

from amt_apc.models import Pipeline


# Create a pipeline without model for feature extraction
_PIPELINE = Pipeline(no_model=True)
_CONFIG = _PIPELINE.config


def wav2feature(path_input: str) -> torch.Tensor:
    """
    Convert a wav file to a feature:
    mel-spectrogram according to config.json

    Args:
        path_input (str): Path to the input wav file.

    Returns:
        torch.Tensor: Feature tensor. (n_frames, n_mels)
    """
    return _PIPELINE.wav2feature(path_input)


def preprocess_feature(feature: torch.Tensor) -> torch.Tensor:
    feature = np.array(feature, dtype=np.float32)

    tmp_b = np.full([_CONFIG["input"]["margin_b"], _CONFIG["feature"]["n_bins"]], _CONFIG["input"]["min_value"], dtype=np.float32)
    len_s = int(np.ceil(feature.shape[0] / _CONFIG["input"]["num_frame"]) * _CONFIG["input"]["num_frame"]) - feature.shape[0]
    tmp_f = np.full([len_s+_CONFIG["input"]["margin_f"], _CONFIG["feature"]["n_bins"]], _CONFIG["input"]["min_value"], dtype=np.float32)

    preprocessed_feature = torch.from_numpy(np.concatenate([tmp_b, feature, tmp_f], axis=0))

    return preprocessed_feature
