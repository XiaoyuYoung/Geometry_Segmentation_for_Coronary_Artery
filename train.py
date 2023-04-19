from trainer import Trainer


def get_yaml(file):
    import yaml
    with open(file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train():
    '''
    注意：
    1. 在Windows测试时,num_worker不可用，所以需要设置为0,否则dataloader过不去
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='./config/config-s1-train.yaml', help="config file",
                        required=True)

    config = get_yaml(parser.parse_args().config)

    trainer = Trainer(config=config)
    trainer.run(DEBUG=True)


if __name__ == '__main__':
    train()
