from .retinaface.retinaface import RetinaFace


def init_detection_model(model_name, half=False): #, device='cuda'):
    if 'retinaface' in model_name:
        model = init_retinaface_model(model_name, half) #, device)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    return model


def init_retinaface_model(model_name, half=True):
    bmodel_path = './bmodel_files/retinaface_F16.bmodel'
    model = RetinaFace(model_path=bmodel_path, half=half)
    return model
