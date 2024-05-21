model_path = {
    "awportraitv14": {
        "name": "awportrait",
        "encoder": "sdv15_text.bmodel",
        "unet": "sdv15_unet_multisize.bmodel",
        "vae_decoder": "sdv15_vd_multisize.bmodel",
        "vae_encoder": "sdv15_ve_multisize.bmodel",
        "latent_shape": {
            "512x512": "True",
            "768x512": "False",
            "512x768": "False"
        }
    }
}
