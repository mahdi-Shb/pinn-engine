# منابع و فایل‌های مرجع همکاران

کد اجرایی از این پوشه خارج شده است:

- GeoCore: `src/geocore`؛ پکیج مستقل با همان `import geocore`.
- شبکه‌ها: `src/pinn_engine/networks`؛ استفاده با `from pinn_engine.networks import ...`.
- قرارداد شبکه‌ها: `src/pinn_engine/networks/_contract.py`؛ رپر GeoCore: `src/pinn_engine/geocore.py`.

فایل `sources.json` commit/آرشیو مبنا و مسیر فعلی کدها را ثبت می‌کند. مجوزها حفظ شده‌اند و همراه بسته نصب می‌شوند.

پوشه‌های باقی‌ماندهٔ `geocore/` و `neural3dbench/` مستندات، مثال‌ها، تست‌ها، تنظیمات و ابزارهای benchmark اصلی را برای مرجع نگه می‌دارند. pyprojectهای داخل آن‌ها metadata پروژهٔ اولیه‌اند و پس از انتقال سورس، مسیر نصب نیستند. برای نصب کتابخانه از pyproject ریشه استفاده شود:

```powershell
python -m pip install -e ".[dev,geometry]"
```

API قدیمی مثال‌ها و تست‌های upstream ممکن است با قرارداد فعلی Model یکسان نباشد. تست‌های فعال موتور در `tests/`، نمونه‌های قابل اجرای موتور در `examples/` و راهنمای فعلی شبکه‌ها در [راهنمای شبکه‌ها](../docs/networks.fa.md) هستند. سورس عددی، خروجی‌های auxiliary و ابزارهای مرجع حذف نشده‌اند.
