def model_loader(model_name):
    if model_name == "ai-forever/Real-ESRGAN":
        from RealESRGAN import RealESRGAN

        model = RealESRGAN("cpu", scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)

        return model.model
    else:
        raise ValueError(f"Invalid model id provided: {model_name}")
