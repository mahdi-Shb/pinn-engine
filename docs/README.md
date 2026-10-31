# راهنمای مستندات موتور جدید

## قرارداد اجرایی فعلی

- [Signature و گروه محورهای ...](signature.fa.md): دریافت مستقیم Value و قرارداد استقلال محورهای هم‌نام.

- [رپر GeoCore](geocore.fa.md): ساخت ترتیبی سناریوها و خروجی‌های عددی قابل اتصال به Solver.
- [شبکه‌های Neural3DBench](networks.fa.md): نگاشت inputs/outputs، محورهای هر شبکه، BatchNorm و بسته‌بندی meshها.
- [قرارداد کمیت، مدل، محورها و مشتق](contracts.fa.md): رفتار پیاده‌شده و محدودیت‌های فعلی.
- [گروه‌بندی درخواست‌های مشتق](derivative_grouping.fa.md): معیارهای reverse، مسیر توسعهٔ forward/Taylor و اشتراک عملگر mesh.
- [محور بیرونی لیستی](tensor_lists.fa.md): نمایش اندازه‌های متفاوت، اجرای حلقه‌ای و vmap و بررسی سازگاری هنگام اجرا.

## یادداشت‌های طراحی

این فایل‌ها تاریخچهٔ بحث و پیشنهادها را نگه می‌دارند؛ همهٔ پیشنهادهای آن‌ها API فعلی نیستند.

- [یادداشت طراحی، تاریخچه و طرح‌های توسعه](architecture_open_decisions.fa.md)
- [طرح توسعهٔ multi-domain geometry، partial AD و کنترل گراف](architecture_open_decisions.fa.md#multi-domain-geometry-and-partial-ad): ایده‌های نسبتاً مشخص برای پیاده‌سازی بعدی، همراه با تجربه و محدودهٔ branch محفوظ؛ این‌ها تصمیم‌های معوق نیستند.
- [پیشنهاد قرارداد batch](model_batch_contract_proposal.fa.md)
- [پیشنهاد حداقلی sampler](sampler_minimal_proposal.fa.md)
- [بررسی پژوهشی وابستگی نقاط و مشتق‌گیری](research/same_coordinate_coupling.fa.md)

## زمینهٔ بازطراحی

دو گزارش زیر دربارهٔ کتابخانهٔ قدیمی PINNlab هستند و برای حفظ دلایل بازطراحی آمده‌اند؛
یافته‌های آن‌ها به معنی وجود همان مشکلات در این موتور نیست. مسیرهای کد قدیمی در
آن‌ها مرجع تاریخی‌اند؛ آن کدها جزو این ریپو نیستند.

- [گزارش محاسبات تکراری](background/computation_audit.fa.md)
- [دسته‌بندی علت‌های طراحی](background/computation_audit_design.fa.md)
