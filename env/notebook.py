import os


def get_env() -> str:
    """check notebook environment

    Returns:
        [str]: Environment type ('KAGGLE' | 'COLAB' | 'LOCAL')
    """
    is_kaggle_env = "KAGGLE_URL_BASE" in set(os.environ.keys())
    is_colab_env = "COLAB_GPU" in set(os.environ.keys()) or "COLAB_TPU" in set(
        os.environ.keys()
    )
    if is_kaggle_env:
        return "KAGGLE"
    elif is_colab_env:
        return "COLAB"
    else:
        return "LOCAL"
