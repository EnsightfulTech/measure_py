from PySide2.QtCore import QSettings
from pathlib import Path

# default paths
INSTALL_PATH = Path(__file__).absolute().parent
CONFIG_PATH = Path(__file__).absolute().parent / "config.ini"
ROOT_DIR = Path.home() / "DCIM"
CAM_PATH = ROOT_DIR / "camera_model.json"
ICON_PATH = Path(__file__).absolute().parent / "icons"

supported_formats = "*.[jpb][npm][gep]*"
supported_formats_qt = "Images (*.png *.jpg *.jpeg *.bmp)"

class ConfigFile:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.settings = QSettings(
            str(self.config_path), 
            QSettings.IniFormat
        )

    def write_config(self):
        self.settings.setValue("button_size", 64)

        self.settings.setValue("Camera/exposure", 20000)
        self.settings.setValue("Camera/iso", 400)

        self.settings.setValue("Measure/default_snap", int(True))
        self.settings.setValue("Measure/pick_integer", int(False))
        self.settings.setValue("Measure/manual_match", int(False))
        self.settings.setValue("scaling/factor", 1.0)
        self.settings.setValue("standard/width", 1920.0)

    def get_config(self, key):
        return int(self.settings.value(key))

    def set_config(self, key, data):
        self.settings.setValue(key, data)
    

if __name__ == "__main__":
    config = ConfigFile()
    config.write_config()
    print(config.get_config("button_size"))
    print(config.get_config("Camera/exposure"))
    print(config.get_config("Camera/iso"))
    print(config.get_config("Measure/default_snap"))
    print(config.get_config("Measure/pick_integer"))
    print(config.get_config("Measure/manual_match"))