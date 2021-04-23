import yaml


def read_config(parts: str) -> dict:
    """

    Returns:
        object: dict
    """
    with open('configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config[parts]
