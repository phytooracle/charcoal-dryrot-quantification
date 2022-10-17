def load_checkpoint(checkpoint, model):
    """[summary]
        Load a saved model
    Args:
        checkpoint ([type]): checkpoint to load weights from
        model ([type]): model to load weights from
    """
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Model Loaded")
