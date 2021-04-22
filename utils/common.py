import yaml

def read_config(parts: str)->dict:
    with open('configs.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    return config[parts]