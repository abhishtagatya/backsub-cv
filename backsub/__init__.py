from backsub.util import load_image


class BaseBSCV:

    def __init__(self, image, background_image, resize=(720, 400), kSize=None, iterC=None):
        self.image = load_image(image, resize)
        self.background_image = load_image(background_image, resize)
        self.resize = resize
        self.kSize = kSize
        self.iterC = iterC

    def process(self):
        return

    def __str__(self):
        return f'{self.__class__.__name__}.{self.kSize}-{self.iterC}'
