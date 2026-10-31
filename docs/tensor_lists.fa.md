# محور بیرونی لیستی و Tensorهای با اندازهٔ متفاوت

این قابلیت یک قرارداد عمومی برای batchهای با اندازهٔ متفاوت است؛ Geometry، Field یا domain
در main نداریم. نام محور بیرونی دلخواه است و به‌خودی‌خود معنای هندسی ندارد. padding یا
job موازی خودکار اضافه نشده است.

## مقدار و امضای تابع

```python
from pinn_engine import Model, Quantity, Signature, Value
import torch

coordinates = Quantity("coordinates", (3,))
embedding = Quantity("embedding", (3,))

class Encoder(Model):
    signature = Signature({coordinates: "point"}, {embedding: ()})

    def evaluate(self, values):
        # One case per call; point counts may differ between cases.
        return {embedding: values[coordinates].mean(dim=0)}

points = [torch.randn(20, 3), torch.randn(40, 3)]
result = Encoder()({coordinates: Value(points, ("case", "point"))})
# result[embedding].tensor is a list of two Tensors, each shaped (3,).
# result[embedding].axes == ("case",)
```

Value.tensor یک Tensor یا لیست غیرخالی و غیرتودرتوی Tensorها می‌پذیرد. محور اول axes متعلق
به لیست است؛ سایر محورهای نمونه در هر Tensor قرار دارند و ابعاد ذاتی Quantity بعد از آن‌ها
هستند. تمام اعضا نام محور و rank یکسان دارند؛ طول محورهای داخلی می‌تواند متفاوت باشد.
داده کپی یا detach نمی‌شود و grad حفظ می‌شود. هنگام اجرا ساختار لیست و metadata را تغییر ندهید.

چند Quantity می‌توانند لیست‌های متناظر داشته باشند. یک اندیس به یک عضو متناظر اشاره می‌کند.
محورهای داخلی point و cell می‌توانند اندازه‌های متفاوت داشته باشند؛ محورهای هم‌نام باید در
هر عضو سازگار باشند. تطابق نام‌ها توسط کاربر اعلام می‌شود.

## اجرای خودکار

- محور بیرونی خارج از Signature: موتور عضوها را با حلقه انتخاب می‌کند، ورودی‌های dense
  دارای همان محور را برش می‌دهد و ورودی‌های فاقد آن را مشترک نگه می‌دارد. داخل هر عضو
  batching معمول اجرا می‌شود. خروجی لیست می‌ماند؛ اعضای هم‌شکل هم خودکار stack نمی‌شوند.
- محور بیرونی داخل Signature: باید اولین محور core ورودی لیستی باشد؛ evaluate خود لیست را
  دریافت می‌کند. محورهای اضافهٔ مشترک می‌توانند با vmap پردازش شوند.
- مثلاً لیست Tensorهای (S, N_i, 3) با محورهای (case, scenario, point) و امضای (case, point)
  باعث می‌شود evaluate لیست (N_i, 3) را ببیند و scenario خودکار vmap شود. S باید مشترک باشد.
- اگر لیست کامل نگه داشته شود و محور point با طول متفاوت باید vmap شود، خطا می‌دهیم؛ تابع
  می‌تواند هر دو محور را کامل مصرف کند و اندازه‌های متفاوت را خودش مدیریت کند.
- خروجی لیستی نیز اولین محور core خروجی را نمایندگی می‌کند. تولید اندازهٔ متغیر زیر vmap
  همچنان محدودیت‌های خود PyTorch را دارد.

## محدودهٔ فعلی

در مسیر batching خودکار، یک محور بیرونی لیستی در هر فراخوانی پشتیبانی می‌شود. در [حالت Signature دارای ...](signature.fa.md)، خود تابع Valueها را مدیریت می‌کند و این محدودیت اجرایی اعمال نمی‌شود. لیست تودرتو، padding و stacking خودکار،
sampler، mesh generator یا زمان‌بندی موازی نداریم. Solver مقدارها را در همان فراخوانی میان
مدل‌ها عبور می‌دهد. AD مستقیم از مشتق‌های تودرتو روی همین لیست‌ها نیز پشتیبانی می‌کند؛
[قرارداد مشتق](contracts.fa.md) جزئیات را توضیح می‌دهد. تست compile با backend eager تضمین
سرعت یا سازگاری Inductor نیست.
