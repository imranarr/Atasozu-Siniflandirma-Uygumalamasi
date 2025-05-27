import sys
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QTextEdit, QPushButton, QLabel, QComboBox,
                            QHBoxLayout, QMessageBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
from model import AtasozuModel
import os
import torch
import threading
from difflib import SequenceMatcher

class AtasozuUygulamasi(QMainWindow):
    model_egitildi = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atasözleri Sınıflandırma ve Anlam Tahmini")
        self.setGeometry(100, 100, 1000, 800)
        
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Giriş alanı
        self.giris_label = QLabel("Atasözünü Giriniz:")
        layout.addWidget(self.giris_label)
        
        self.giris_text = QTextEdit()
        self.giris_text.setPlaceholderText("Atasözünü buraya yazın...")
        layout.addWidget(self.giris_text)
        
        # Tahmin butonu
        self.tahmin_buton = QPushButton("Tahmin Et")
        self.tahmin_buton.clicked.connect(self.tahmin_yap)
        layout.addWidget(self.tahmin_buton)
        
        # Sonuç alanları
        self.kategori_label = QLabel("Tahmin Edilen Kategori:")
        layout.addWidget(self.kategori_label)
        
        self.kategori_sonuc = QLabel("")
        layout.addWidget(self.kategori_sonuc)
        
        self.anlam_label = QLabel("Tahmin Edilen Anlam:")
        layout.addWidget(self.anlam_label)
        
        self.anlam_sonuc = QLabel("")
        self.anlam_sonuc.setWordWrap(True)
        layout.addWidget(self.anlam_sonuc)
        
        # Yeni atasözü ekleme bölümü
        layout.addWidget(QLabel("\nYeni Atasözü Ekle:"))
        
        # Yatay layout için
        input_layout = QHBoxLayout()
        
        # Atasözü girişi
        self.yeni_atasozu = QLineEdit()
        self.yeni_atasozu.setPlaceholderText("Yeni atasözü")
        input_layout.addWidget(self.yeni_atasozu)
        
        # Kategori seçimi
        self.kategori_combo = QComboBox()
        self.kategori_combo.addItems(self.get_categories())
        input_layout.addWidget(self.kategori_combo)
        
        # Anlam girişi
        self.yeni_anlam = QLineEdit()
        self.yeni_anlam.setPlaceholderText("Anlamı")
        input_layout.addWidget(self.yeni_anlam)
        
        layout.addLayout(input_layout)
        
        # Ekle butonu
        self.ekle_buton = QPushButton("Yeni Atasözü Ekle")
        self.ekle_buton.clicked.connect(self.yeni_atasozu_ekle)
        layout.addWidget(self.ekle_buton)
        
        # Model yükle
        self.model_yukle()
        
        self.model_egitildi.connect(self.egitim_bitti_mesaj)
    
    def get_categories(self):
        try:
            with open('data/atasozleri.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'atasozleri' in data:
                    return list(set(item['kategori'] for item in data['atasozleri'] if 'kategori' in item))
                else:
                    return []
        except Exception as e:
            print(f"Error reading categories: {str(e)}")
            return []
    
    def model_yukle(self):
        try:
            self.model = AtasozuModel()
            if not os.path.exists('model_weights.pt'):
                def train_and_save():
                    self.model.train()
                    torch.save(self.model.model.state_dict(), 'model_weights.pt')
                    self.model_egitildi.emit()
                threading.Thread(target=train_and_save, daemon=True).start()
            else:
                self.model.model.load_state_dict(torch.load('model_weights.pt'))
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model yüklenirken hata oluştu: {str(e)}")
    
    def tahmin_yap(self):
        atasozu = self.giris_text.toPlainText().strip()
        if not atasozu:
            self.kategori_sonuc.setText("Lütfen bir atasözü girin!")
            self.anlam_sonuc.setText("")
            return
        
        try:
            # Tahmin yap
            tahmin_kategori = self.model.predict(atasozu)
            
            # Tüm atasözleri arasında en benzerini bul
            with open('data/atasozleri.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            en_yakin_anlam = "Bu atasözü için anlam tahmini yapılamadı."
            max_benzerlik = 0
            for item in data['atasozleri']:
                benzerlik = SequenceMatcher(None, atasozu, item.get('metin', '')).ratio()
                if benzerlik > max_benzerlik:
                    max_benzerlik = benzerlik
                    en_yakin_anlam = item.get('anlam', '')
            
            # Sonuçları göster
            self.kategori_sonuc.setText(tahmin_kategori)
            self.anlam_sonuc.setText(en_yakin_anlam)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin yapılırken hata oluştu: {str(e)}")
    
    def yeni_atasozu_ekle(self):
        yeni_atasozu = self.yeni_atasozu.text().strip()
        kategori = self.kategori_combo.currentText()
        anlam = self.yeni_anlam.text().strip()
        
        if not all([yeni_atasozu, kategori, anlam]):
            QMessageBox.warning(self, "Uyarı", "Lütfen tüm alanları doldurun!")
            return
        
        try:
            with open('data/atasozleri.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Yeni atasözünü ekle
            data['atasozleri'].append({
                "metin": yeni_atasozu,
                "kategori": kategori,
                "anlam": anlam
            })
            
            # Dosyaya kaydet
            with open('data/atasozleri.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # Modeli yeniden eğit
            self.model.train()
            torch.save(self.model.model.state_dict(), 'model_weights.pt')
            
            # Alanları temizle
            self.yeni_atasozu.clear()
            self.yeni_anlam.clear()
            
            QMessageBox.information(self, "Başarılı", "Yeni atasözü başarıyla eklendi!")
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Atasözü eklenirken hata oluştu: {str(e)}")

    def egitim_bitti_mesaj(self):
        QMessageBox.information(self, "Model Eğitimi", "Model eğitimi tamamlandı!")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = AtasozuUygulamasi()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        print('Uygulama başlatılırken hata oluştu:')
        traceback.print_exc() 