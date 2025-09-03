# VisionSuite Core — Synthetic Data Training from a Single Photo

---

## 🇹🇷 Türkçe

### 🎯 Amaç
Endüstride birçok senaryoda ürünlerin **konum ve yönelim açısı** kritik öneme sahiptir.  
Fakat geniş ölçekli **etiketlenmiş veri** toplamak maliyetlidir.  

Bu projede hedefimiz:  
- **Tek bir referans fotoğraf** ile ürünü tanıtmak,  
- Kullanıcının bu fotoğraf üzerinde ürün bölgesini (polygon/dikdörtgen) ve **grip + yön** noktalarını işaretlemesini sağlamak,  
- Buradan **sentetik veri seti** üretmek,  
- **YOLOv8** modeli ile otomatik eğitim yapmak,  
- Eğitilen model ile ürünü gerçek zamanlı olarak **konum + açı bilgisi** ile takip etmek.  

---

### ⚙️ Mimari
- **Pylon Kamera** → canlı görüntü alımı  
- **Template üretimi** → ilk fotoğraf üzerinde polygon/rect çizimi + grip/dir seçimi  
- **Sentetik veri üretimi** → ürün rastgele açılar ve arka planlarla çoğaltılır  
- **YOLOv8 eğitimi** → otomatik detect/segment/pose modu algılanır  
- **Takip** → ürünün merkez koordinatları ve açısı gerçek zamanlı hesaplanır  

---

### 📦 Kurulum
```bash
git clone https://github.com/<kullanici>/<repo>.git
cd <repo>
pip install -r requirements.txt

### 📂 Klasör Yapısı
VisionSuite/data/
  datasets/product_<id>/{images,labels}/{train,val}
  models/product_<id>.pt
  templates/product_<id>/{template.png, template.json}
  calibration.jpg
