
def preprocess_image(image):
    crop_x1 = 0
    crop_x2 = 320
    crop_y1 = 60
    crop_y2 = 140
    return image[crop_y1:crop_y2,
                 crop_x1:crop_x2]
