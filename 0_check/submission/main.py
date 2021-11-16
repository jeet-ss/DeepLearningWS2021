from pattern import *
from generator import ImageGenerator

if __name__ == "__main__":
    checker=Checker(200, 10)
    #checker.show()

    cir=Circle(100, 20, (50,40))
    cir.show()

    sp=Spectrum(100)
    #sp.show()

    label_path = './Labels.json'
    file_path = './exercise_data/'
    #gen = ImageGenerator(file_path, label_path, 5, [32, 32, 3], rotation=False, mirroring=True, shuffle=True)
    #gen.show()