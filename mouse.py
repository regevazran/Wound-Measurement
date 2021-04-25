

class Mouse:
    def __init__(self):
        self.name = None
        self.contraction = None
        self.scab = None
        self.wound_close = None
        self.wound_size_by_pixel = None
        self.wound_size_by_cm = None
        self.pictures = None

    def add_name(self, mouse_name, exp_name):
        self.name = str(exp_name) + "_" + str(mouse_name)

    def add_day(self, day, contraction, scab, wound_close, wound_size_by_pixel, wound_size_by_cm, pictures):
        self.contraction[day] = contraction
        self.scab[day] = scab
        self.wound_close[day] = wound_close
        self.wound_size_by_pixel[day] = wound_size_by_pixel
        self.pictures[day] = pictures
        self.wound_size_by_cm[day] = wound_size_by_cm

