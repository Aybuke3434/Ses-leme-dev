# Ses İşleme Dersi Ödev 

Input: Modelin giriş katmanı. Veri bu katman aracılığıyla modele aktarılır.

GRU (Gated Recurrent Unit): Giriş katmanı iki farklı GRU katmanına ayrılır. GRU, LSTM gibi bir tür tekrarlayan sinir ağıdır ancak daha az karmaşıktır ve hesaplama olarak daha verimlidir. İki farklı GRU katmanı paralel olarak kullanılmış.

ReLU (Rectified Linear Unit): GRU katmanlarından çıkan veriler ReLU aktivasyon fonksiyonundan geçirilir. Bu fonksiyon, negatif değerleri sıfıra çeker ve pozitif değerleri olduğu gibi bırakır. Modelin doğrusal olmayan karmaşıklık seviyesini artırmak için kullanılır.

LSTM (Long Short-Term Memory): ReLU katmanlarından çıkan veriler LSTM katmanına verilir. LSTM, uzun süreli bağımlılıkları öğrenmede oldukça etkilidir. GRU'dan farklı olarak, daha karmaşık bir hücre yapısına sahiptir.

ReLU: LSTM katmanından çıkan veriler tekrar ReLU aktivasyon fonksiyonundan geçirilir.

Conv1D (1D Convolution): Veriler Conv1D katmanlarına paralel olarak ayrılır. Conv1D, özellikle zaman serisi verileri veya bir boyutlu veriler için kullanılır. Her iki Conv1D katmanında da ReLU aktivasyon fonksiyonu kullanılmıştır.

Dense: Conv1D katmanlarından çıkan veriler birleştirilir ve tam bağlı bir Dense katmanına verilir. Dense katmanı, tüm girdiler ile tüm çıktılar arasında bağlantı kurarak çalışır.

Output: Modelin çıkış katmanı. Dense katmanından çıkan veriler, modelin nihai çıktısını oluşturur.
