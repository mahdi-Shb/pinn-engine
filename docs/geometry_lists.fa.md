# روش اول: محور بیرونی لیستی برای geometryها

این روش بدون padding و بدون job موازی پیاده شده است. Quantity همچنان شکل ذاتی عددی را معرفی می‌کند؛ تعداد geometry، رأس و سلول در آن نوشته نمی‌شود. ارتباط هندسی میان domainها تغییر نکرده است.

## مقدار و امضای تابع

```python
from pinn_engine import Model, Quantity, Signature, Value
import torch

coordinates = Quantity("coordinates", (3,))
embedding = Quantity("embedding", (3,))

class Encoder(Model):
    geometry = None
    signature = Signature({coordinates: "point"}, {embedding: ()})

    def evaluate(self, values):
        # One geometry per call; point counts may differ between geometries.
        return {embedding: values[coordinates].mean(dim=0)}

points = [torch.randn(20, 3), torch.randn(40, 3)]
result = Encoder()({coordinates: Value(points, ("geometry", "point"))})
# result[embedding].tensor is a list of two Tensors, each shaped (3,).
# result[embedding].axes == ("geometry",)
```

Value.tensor اکنون Tensor یا لیست غیرخالی و غیرتودرتوی Tensorها می‌پذیرد. محور اول axes متعلق به لیست است؛ سایر محورهای نمونه در هر Tensor قرار دارند و ابعاد ذاتی Quantity بعد از آن‌ها هستند. نام محور بیرونی دلخواه است. تمام اعضا نام محور و rank یکسان دارند؛ طول محورهای داخلی می‌تواند متفاوت باشد. داده کپی یا detach نمی‌شود و grad حفظ می‌شود. لیست و storage تنسورها قابل تغییرند؛ هنگام استفاده، ساختار و metadata را تغییر ندهید.

مختصات و اتصال‌ها می‌توانند دو Quantity با دو لیست متناظر باشند. اندیس یکسان باید همان geometry را مشخص کند؛ نام محور point و cell مستقل است و تعدادشان لازم نیست برابر باشد. هر محور هم‌نام بین مقادیر مختلف باید در هر geometry طول سازگار داشته باشد. نام یکسان اعلام تطابق توسط کاربر است، نه اثبات هندسی آن.

## اجرای خودکار

- geometry خارج از Signature: موتور آن را با حلقه انتخاب می‌کند، ورودی‌های dense دارای همان محور را هم برش می‌دهد و ورودی‌های فاقد آن را مشترک نگه می‌دارد. سپس در هر geometry، batching معمول اجرا می‌شود. خروجی‌ها با محور بیرونی لیستی جمع می‌شوند؛ حتی اگر هم‌شکل باشند خودکار stack نمی‌شوند.
- geometry داخل Signature: این محور باید برای ورودی لیستی اولین محور core باشد و evaluate خود لیست را دریافت می‌کند. محورهای اضافی Tensorهای داخل لیست در صورت طول مشترک با vmap پردازش می‌شوند. خروجی لیستی نیز اولین محور core خروجی را نمایندگی می‌کند؛ محورهای batch بعد از محور لیست قرار می‌گیرند.
- برای مثال لیست Tensorهایی با شکل (S, N_i, 3) و محورهای (geometry, scenario, point) را می‌توان با امضای ورودی (geometry, point) اجرا کرد: evaluate لیست (N_i, 3) می‌بیند و scenario خودکار vmap می‌شود. S باید میان geometryها برابر باشد.
- اگر geometry کامل دریافت شود اما point با طول‌های ۲۰ و ۴۰ باید پیمایش شود، قبل از evaluate خطا می‌دهیم. دریافت کامل هر دو محور مجاز است؛ تابع خودش می‌تواند تعدادهای مختلف را مدیریت یا داخلاً pad کند.
- مدل می‌تواند لیست تولید کند، مثلاً با امضای خروجی (geometry, point). تولید خروجی متغیر در هر عضو یک vmap همچنان مشمول محدودیت‌های خود PyTorch است؛ پشتیبانی از همهٔ توابع دلخواه ادعا نمی‌شود.

## محدودهٔ فعلی

فقط یک محور بیرونی لیستی در هر فراخوانی مدل پشتیبانی می‌شود. ورودی و خروجی تودرتوی لیستی، padding خودکار، stacking خودکار، mesh generator، sampler و موازی‌سازی jobها اضافه نشده‌اند. بررسی‌ها زمان اجرا و پیش از محاسبه‌اند؛ طراحی بررسی هنگام ساخت Solver هنوز کار ضروری آینده است.

Solver همان Valueها را میان مدل‌ها عبور می‌دهد و نتایج مشترک را دوباره استفاده می‌کند. backend عددی مشتق همچنان پیاده نشده است؛ این تغییر فقط اجرای عادی و autograd آموزش روی Tensorها را پشتیبانی می‌کند. آزمون torch.compile با backend eager فقط ثبت گراف را بررسی می‌کند و تضمین سرعت یا سازگاری Inductor نیست.
