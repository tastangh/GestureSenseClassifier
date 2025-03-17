import torch
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    """
    EMG verilerini model ile kullanılmak üzere dönüştürmek için kullanılan bir veri kümesi sınıfı.
    
    Bu sınıf, PyTorch'un Dataset sınıfından türetilmiştir ve modelin eğitimi için gerekli özellikler ve etiketleri sağlar.
    """
    
    def __init__(self, features, labels):
        """
        EMGDataset sınıfının yapıcı metodu. Özellikleri ve etiketleri alır ve PyTorch tensörlerine dönüştürür.
        
        Parametreler:
        ------------
        features : pandas.DataFrame veya numpy.ndarray
            Modelin eğitimi için kullanılacak özellik verileri.
        
        labels : pandas.Series veya numpy.ndarray
            Modelin eğitimi için kullanılacak etiket verileri.
        
        Yorumlar:
        ---------
        - `features` verisi, PyTorch modeline uygun hale getirilmek için `torch.float32` tipine dönüştürülür.
        - `unsqueeze(-1)` metodu ile özelliklerin son boyutuna ekstra bir boyut eklenir. Bu, 1D konvolüsyon katmanları için gerekli olabilir.
        - `labels` verisi, `torch.long` tipine dönüştürülerek sınıflandırma görevleri için uygun hale getirilir.
        """
        # Özellikleri PyTorch tensörüne dönüştür ve son boyutuna bir boyut ekle
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(-1)
        
        # Etiketleri PyTorch tensörüne dönüştür
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self):
        """
        Veri kümesinin uzunluğunu döndürür.
        
        Returns:
        --------
        int
            Veri kümesindeki örnek sayısı.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Belirtilen indeksdeki özellik ve etiket çiftini döndürür.
        
        Parametreler:
        ------------
        idx : int
            Veri kümesindeki örneğin indeksi.
        
        Returns:
        --------
        tuple
            - features : torch.Tensor
                Belirtilen indeksdeki özellik verisi.
            - label : torch.Tensor
                Belirtilen indeksdeki etiket verisi.
        
        Yorumlar:
        ---------
        - `idx` parametresi, veri kümesindeki hangi örneğin alınacağını belirtir.
        - Fonksiyon, belirtilen indeksteki özellik ve etiket çiftini bir demet (tuple) olarak döndürür.
        """
        return self.features[idx], self.labels[idx]