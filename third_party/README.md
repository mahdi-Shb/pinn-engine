# کدهای همکاران برای تطبیق با موتور

این پوشه سورس دو پروژهٔ همکاران را نگه می‌دارد. شبکه‌های Neural3DBench اکنون مستقیماً Model موتور هستند و در نصب موتور با نام `pinn_engine.neural3dbench` ارائه می‌شوند؛ سورس عددی جابه‌جا یا کپی نشده است. GeoCore همچنان dependency اختیاری با نصب جداگانه است و رپر موتور در `src/pinn_engine/geocore.py` قرار دارد.

- `geocore/`: محتوای پروژهٔ geometry از آرشیو ارسالی، شامل سورس، مستندات، تست‌ها، مثال‌ها، داده‌های نمونه و کتابخانه‌های native همراه.
- `neural3dbench/`: فایل‌های ثبت‌شدهٔ ریپوی GitHub، شامل شبکه‌ها، adapterها، تست‌ها، مثال‌ها و نمونه‌خروجی‌ها؛ بدون تاریخچهٔ Git.
- `sources.json`: آدرس و commit منبع شبکه‌ها و SHA-256 آرشیو geometry برای شناسایی نسخهٔ واردشده.

فایل‌های cache، محیط مجازی و خروجی build وارد نشده‌اند. مجوزها و اعلان‌های موجود در منابع حفظ شده‌اند؛ LICENSE شبکه‌ها در wheel موتور هم قرار می‌گیرد. commit موجود در sources.json نسخهٔ مبناست و local_changes تغییرات تطبیقی بعد از import را معرفی می‌کند. هیچ تغییری در `sys.path` لازم نیست.

## وضعیت تطبیق فعلی

هر یازده شبکه inputs/outputs صریح، محورهای core و independence دارند؛ BatchNorm کل batch را می‌گیرد، گراف‌ها درون evaluate به صورت disjoint بسته‌بندی می‌شوند و سایر شبکه‌ها batching خودکار دارند. جزئیات، محدودیت‌ها و روش استفاده در [راهنمای شبکه‌ها](../docs/neural3dbench.fa.md) است.

API قدیمی `network(ModelInput)` به `network.predict(ModelInput)` منتقل شده تا auxiliary/metadata حفظ شوند؛ constructor هم inputs/outputs می‌خواهد. مثال‌ها و تست‌های upstream موجود در این snapshot هنوز API قدیمی را نشان می‌دهند. تست قرارداد جدید در `tests/test_neural3dbench.py` موتور است و مثال قابل اجرا در `examples/geocore_mesh_network.py` موتور قرار دارد.

سه محدودیت باقی است: BatchNorm آمار batch با coreهای ragged پذیرفته نیست؛ مختصات GINO باید واقعاً به `[0,1]` برده شوند؛ هندسهٔ DeepNURBS ثابت و در config است. تولید موازی GeoCore، domain و مشتق‌های تخصصی شبکه‌ها برای آینده‌اند.

خود الگوریتم‌های geometry تغییر نکرده‌اند. هم‌ترازی محورها، انتقال channelها و reshape مؤلفه‌ها در helper قرارداد موتور انجام می‌شود، نه با adapter جدا برای هر فراخوانی.
