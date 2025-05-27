# Atasözü Sınıflandırma Uygulaması

Atasözlerini kategorilere ayırıp anlamlarını tahmin eden basit bir Python uygulaması. BERT tabanlı yapay zeka modeli kullanılarak geliştirilmiştir.

##  Kurulum

1. Python'u yükleyin (Python 3.8 veya üstü)
2. Projeyi indirin
3. `uygulama_baslat.bat` dosyasına çift tıklayın
4. Program otomatik olarak başlayacaktır

##  Kategoriler

Program atasözlerini 5 ana kategoriye ayırır:
- Ahlak ve Değerler (Örn: "Doğru söyleyeni dokuz köyden kovarlar")
- Çalışma ve Başarı (Örn: "Emek olmadan yemek olmaz")
- İlişkiler ve Toplum (Örn: "Komşu komşunun külüne muhtaçtır")
- Bilgelik ve Tecrübe (Örn: "Ateş düştüğü yeri yakar")
- Doğa ve Hayat (Örn: "Mart kapıdan baktırır")

##  Kullanım

1. Programı başlatın
2. Atasözü girin
3. "Tahmin Et" butonuna tıklayın
4. Program size:
   - Atasözünün kategorisini
   - Anlamını
   - Benzer atasözlerini gösterecektir

##  Örnekler

- "Ağaç yaşken eğilir" (Eğitim kategorisi)
- "Bir elin nesi var, iki elin sesi var" (İşbirliği kategorisi)
- "Damlaya damlaya göl olur" (Birikim kategorisi)

##  Teknik Detaylar

- PyQt6 ile geliştirilmiş masaüstü arayüzü
- BERT tabanlı Türkçe dil modeli
- JSON formatında veri depolama
- Otomatik model eğitimi