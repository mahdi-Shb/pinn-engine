# نرمال‌سازی مشترک

منطق عددی در `src/pinn_engine/networks/normalization.py` قرار دارد؛ مستقل از Model و شبکه قابل استفاده است:

```python
import torch
from pinn_engine.networks.normalization import BatchNorm1d

layer = BatchNorm1d(16)
features = [torch.randn(1, 16, 20), torch.randn(1, 16, 40)]
normalized = layer(features)  # Same two shapes; one shared set of statistics.
sum(t.square().mean() for t in normalized).backward()
```

این utility از قرارداد channel-first خود PyTorch استفاده می‌کند؛ ورودی‌های عمومی Model همچنان ابعاد ذاتی انتهایی دارند و adapter قرارداد، تبدیل را انجام می‌دهد.

## BatchNorm

`BatchNorm1d`، `BatchNorm2d` و `BatchNorm3d` از کلاس‌های متناظر PyTorch ارث می‌برند. ورودی Tensor بدون mask مستقیماً از forward اصلی PyTorch می‌گذرد. ورودی list مجموعهٔ قطعات **یک batch** است؛ اعضا می‌توانند تعداد نمونه و ابعاد مکانی متفاوت داشته باشند ولی rank، تعداد کانال، dtype و device مشترک دارند. محور نمونه و تمام محورهای مکانی در آمار هر کانال شریک‌اند؛ هندسهٔ دارای نقاط بیشتر سهم بیشتری دارد، نه اینکه هر هندسه وزن مساوی داشته باشد.

برای mask، آرگومان دوم را بدهید: `layer(tensors, masks)`. ماسک هر عضو بولی و به شکل `(B, *spatial)` است. آمار تنها از نمونه‌های معتبر حساب می‌شود؛ slot نامعتبر در خروجی صفر است و حتی ورودی NaN در این slotها وارد آمار نمی‌شود. شرط حداقل نمونهٔ BatchNorm در training همچنان برقرار است.

پارامترهای affine، running_mean، running_var، num_batches_tracked و momentum دقیقاً به پیاده‌سازی native PyTorch سپرده شده‌اند. یک فراخوانی روی لیست یک بار running statistics را به‌روز می‌کند. میانگین/واریانس، خروجی، گرادیان‌ها و حالت momentum=None با اجرای native روی اجتماع نمونه‌های واقعی مقایسه شده‌اند. این قابلیت چند forward مستقل با running statistics جدا نیست.

## فقط روی point

`InstanceNorm1d` برای هر نمونه و کانال جدا روی نقاط نرمال می‌کند و Tensor/list و mask می‌پذیرد. حالت ماسک‌دار به `track_running_stats=False` نیاز دارد؛ حالت عادی بدون mask رفتار native را حفظ می‌کند، از جمله محدودیت تعداد نقاط آن. برای حالت ماسک‌دار، slotهای نامعتبر صفر می‌شوند و اگر هیچ نقطهٔ معتبری نماند خروجی آن نمونه صفر است؛ ورودی شبکه‌های PointNet همچنان حداقل یک نقطهٔ معتبر می‌خواهد.

PointNet اکنون سه گزینه دارد:

- `normalization="batch"`: رفتار اصلی BatchNorm، با آمار مشترک batch.
- `normalization="instance"`: روی نقاط هر هندسه جدا؛ `affine=True` و بدون running statistics. طبق توافق، بعد از pooling دیگر normalization نداریم؛ در لایه‌های خطیِ بدون normalization بایاس فعال است.
- `normalization="none"`: بدون normalization.

گزینهٔ instance برای PointNet اضافه شده است؛ PointNet++ فعلاً همان گزینه‌های batch و none را دارد. امضای instance شامل `...` نمی‌شود؛ موتور batchهای اضافی را مستقل map می‌کند. استقلال point همچنان False است چون خود normalization و pooling نقاط را به هم وابسته می‌کنند.

## اتصال به شبکه و لیست‌های با اندازهٔ متفاوت

برای PointNet و PointNet++ دارای BatchNorm، `networks/_point_batches.py` ورودی‌های لیستی را موقتاً به یک batch دارای padding تبدیل می‌کند. این padding وارد آمار normalization یا pooling نمی‌شود و پس از اجرا از خروجی حذف می‌شود؛ Value عمومی همچنان لیست است. خود شبکه یک بار اجرا می‌شود و BatchNormهای بعد از pooling تمام embeddingهای واقعی batch را می‌بینند.

ماسک از طریق یک context محلیِ محدود به همان فراخوانی به لایه‌های normalization می‌رسد؛ نیازی به تغییر امضای تک‌تک Convها و MLPها نیست. در PointNet++، mask هنگام تغییر سطح/گروه به همان سطح تبدیل می‌شود. تکرار همسایهٔ واقعی در slotهای اضافی radius grouping مطابق الگوریتم اصلی حفظ می‌شود؛ فقط padding و centroidهای نامعتبر جدید حذف می‌شوند. context پس از خروج یا exception پاک می‌شود و batch/cache را در ماژول نگه نمی‌دارد.

mask صریح در ورودی این دو شبکه نیز اکنون آمار pointwise normalization را محدود می‌کند؛ این اصلاح نسبت به upstream است که mask را فقط در pooling/خروجی اعمال می‌کرد. ورودی‌های نامعتبر پیش از Conv صفر می‌شوند تا NaN نتواند گرادیان وزن‌ها را خراب کند. رفتار dense بدون mask، ترتیب عملیات شبکه، پیش‌فرض batch و کلیدهای state_dict حفظ شده‌اند و با لایه‌های native مقایسه می‌شوند.

این مسیر padding محاسبه و حافظهٔ اضافی دارد؛ هدف آن حفظ معماری اصلی با کمترین تغییر است، نه اجرای packed همهٔ عملیات PointNet. گراف‌ها همچنان با connectivity جدا از هم بسته‌بندی می‌شوند و padding ندارند. BatchNorm3d به‌تنهایی لیست را می‌پذیرد، اما اجرای کامل FNO/U-Net روی gridهای ragged با آمار batch هنوز نیازمند adapter اختصاصی است و خطا می‌دهد؛ padding عمومی grid می‌تواند معنی FFT یا مرزهای convolution را عوض کند.

تست‌ها روی CPU شامل parity با native، آمار بدون padding، شمارنده‌ها، mask/NaN، gradcheck و gradgradcheck، vmap برای instance، آموزش ragged با T-Netها، و برابری خروجی/گرادیان/بافرها پس از افزودن padding هستند. از این تست‌ها تضمین GPU یا fullgraph compilation مسیر masked/list نتیجه نمی‌شود.
