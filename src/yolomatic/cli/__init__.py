# CLI entrypoint package
# Note: Each entrypoint imports directly from its module to avoid loading unnecessary dependencies
# e.g., yolomatic-tensorboard imports from tensorboard_launcher without triggering torch import from run.py
