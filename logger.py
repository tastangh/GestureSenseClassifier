import sys
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        """
        Logger sınıfı, hem terminal çıktısını hem de log dosyasını yönetir.
        :param log_dir: Log dosyalarının kaydedileceği klasör
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
        self.terminal = sys.stdout
        self.log = open(self.log_file, "a")

    def write(self, message):
        """
        Yazılacak mesajı hem terminale hem de log dosyasına yönlendirir.
        :param message: Yazılacak mesaj
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Python 3 uyumluluğu için flush metodu.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """
        Log dosyasını kapatır.
        """
        self.log.close()

# Logger'ı etkinleştirme
def setup_logger():
    """
    Logger'ı sistemin stdout'una bağlar.
    """
    logger = Logger()
    sys.stdout = logger
    return logger
