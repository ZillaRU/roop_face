import numpy as np
from PIL import Image
import sys
from codeformer_model import setup_model

face_restorer = setup_model()
original_image = Image.open('/data/SadTalker/examples/source_image/art_7.png')
print(f"Restore face with Codeformer")
numpy_image = np.array(original_image)
numpy_image = face_restorer.restore(numpy_image)
restored_image = Image.fromarray(numpy_image)
restored_image.save('/data/SadTalker/examples/after_restore.png')
result_image = Image.blend(
    original_image, restored_image, 1.0 #upscale_options.restorer_visibility
)
result_image.save('/data/SadTalker/examples/after_restore_full.png')