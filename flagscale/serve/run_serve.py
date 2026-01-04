from flagscale.serve.arguments import parse_config
from flagscale.serve.engine import ServeEngine


def main():
    config = parse_config()
    engine = ServeEngine(config)
    engine.run_task()


if __name__ == "__main__":
    main()
