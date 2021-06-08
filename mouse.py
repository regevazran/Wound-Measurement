class Mouse:
    def __init__(self):
        self.name = ""
        self.contraction = False
        self.scab = None
        self.wound_close = None
        self.wound_size_by_pixel = None
        self.wound_size_by_cm = None
        self.pictures = None

    def add_name(self, mouse_name, exp_name):
        self.name = str(exp_name) + "_" + str(mouse_name)

    def set_data(self, contraction, scab=None, wound_close=None, wound_size_by_pixel=None, wound_size_by_cm=None, pictures=None):
        self.contraction = contraction
        self.scab = scab
        self.wound_close = wound_close
        self.wound_size_by_pixel = wound_size_by_pixel
        self.pictures = pictures
        self.wound_size_by_cm = wound_size_by_cm
