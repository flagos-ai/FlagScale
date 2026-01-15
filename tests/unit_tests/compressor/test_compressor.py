from flagscale.compress.compressor import compress, prepare_config


def test_config():
    # test_config_path = "test_config.yaml"
    cfg = prepare_config()
    compress(cfg)
