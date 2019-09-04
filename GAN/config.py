class Config:
    def __init__(self):
        self.img_shape = (1, 28, 28)
        self.img_path = ''
        self.discriminator_path = 'model/d.model'
        self.generator_path = 'model/g.model'
        self.z_size = 128