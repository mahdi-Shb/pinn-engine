# رپر GeoCore

`GeoCoreModel` یک Model موتور است و خود سورس GeoCore را تغییر نمی‌دهد. برای نصب:

```powershell
python -m pip install -e ".[dev]"
python -m pip install -e third_party/geocore
python examples/geocore_design.py
```

ورودی `source` یک Design یا هندسهٔ ثابت GeoCore است. `parameters` نام پارامتر طراحی را به Quantity اسکالر وصل می‌کند. در زمان اجرا هر مقدار باید دقیقاً یک محور سناریو داشته باشد؛ اسم آن با `scenario_axis` تعیین می‌شود و مقدار می‌تواند بردار Tensor یا لیست Tensor اسکالر باشد. کل محور در Signature دریافت می‌شود و evaluate با حلقه، `with_params` را برای هر عضو اجرا می‌کند؛ از vmap روی عملیات NumPy استفاده نمی‌شود. template تغییر نمی‌کند. بدون parameters، یک سناریو از هندسهٔ ثابت تولید می‌شود.

`representation` یکی از MESH_GRAPH، POINT_CLOUD، VOXEL_GRID، PARAMETRIC_SURFACE یا PINN_COLLOCATION است؛ aliasهای خود GeoCore مانند PINN هم پذیرفته‌اند. `outputs` نگاشت نام آرایهٔ GeoCore به Quantity کاربر است و فقط خروجی‌های عددی مشخص را می‌پذیرد. `options` تنظیمات builder مانند resolution، تعداد نقاط، normalization و channels است. source، options و Quantityها پس از ساخت مدل تغییر داده نشوند.

تمام خروجی‌ها لیست‌اند و محور اولشان scenario_axis است؛ حتی اعضای هم‌اندازه نیز خودکار stack نمی‌شوند. هر سناریو جدا تولید می‌شود؛ batch داخلی GeoCore برای point cloud، voxel و UV برابر یک است و حذف می‌شود. هیچ padding یا ادغام خودکار گراف‌ها انجام نمی‌شود.

| خروجی | محورهای داخل هر عضو | شکل ذاتی نمونه |
|---|---|---|
| مختصات mesh | vertex | (3,) |
| edge_index | edge | (2,) |
| edge_features | edge | (Fe,) |
| مختصات point cloud | point | (3,) |
| point_mask | point | () |
| voxel_fields | z, y, x | (C,) |
| control_points | control_u, control_v | (2,) |
| UV و physical_coordinates | point | (2,) |
| x_interior و sdf_interior | interior | (3,) و () |
| x_boundary و normal_boundary و tag_boundary | boundary | (3,)، (3,) و () |

edge_index از (2,E) به (E,2) و voxel از (C,D,H,W) به (D,H,W,C) تبدیل می‌شود تا ابعاد ذاتی در انتها باشند. تغییر شکل دلخواه انجام نمی‌شود؛ shape کمیت باید با خروجی برابر باشد. برای جلوگیری از اشتراک نادرست محور بین چند representation، می‌توان با output_axes نام محورهای هر خروجی را تغییر داد؛ مثلاً مختصات گراف و ابرنقاط نام محور مشترک نمی‌خواهند مگر واقعاً نمونه‌های یکسان باشند.

محاسبات و خروجی فعلاً CPU هستند؛ ورودی CUDA یا requires_grad=True صریحاً رد می‌شود. از این wrapper نمی‌توان مشتق نسبت به پارامتر طراحی گرفت. خروجی را بعداً می‌توان به GPU منتقل کرد؛ انتقال بخشی از این کلاس نیست. seed مشخص در هر فراخوانی نمونه‌ها را تکرار می‌کند؛ seed=None نمونه‌های تازه می‌دهد. Integer و bool حفظ می‌شوند و floating_dtype فقط اعشار را تبدیل می‌کند. داده‌های رشته‌ای مثل نام tagها وارد Value نمی‌شوند؛ ترتیب ثابت tagها در Design مسئولیت source است.

انتخاب outputs تنها خروجی عمومی رپر را محدود می‌کند؛ خود GeoCore ممکن است در builder محاسبات دیگر هم انجام دهد. تضمین pruning، compile یا آماده‌سازی موازی نداریم. DataLoader و هر دو حالت RAR مطابق note قبلی برای آینده باز هستند. نصب dependencyهای GeoCore اختیاری است؛ import معمول pinn_engine به NumPy/Numba وابسته نیست.
