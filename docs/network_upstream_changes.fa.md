# تفاوت با شبکه‌های اصلی همکار

منبع، commit `1b635f4d1c3b47fab5c9e9a46b1824b789e478ee` از Neural3DBench است. snapshot اصلی در تاریخچهٔ موتور با `80e5c8d` و کد فعال در `src/PINNCore/networks` قرار دارد.

> معماری‌ها و هستهٔ عددی از کد همکار آمده‌اند. قرارداد Quantity/Value، batching، قواعد استقلال و normalization ماسک‌پذیر تغییرات موتورند. حفظ الگوریتم به معنی یکسان‌بودن همهٔ رفتارهای اجرایی نیست.

| قسمت | قرارداد فعلی |
|---|---|
| کلاس شبکه | هر ۱۱ شبکه مستقیماً Model هستند و signature و evaluate خودشان را دارند؛ BaseNetwork و BaseGraphNetwork حذف شده‌اند. |
| اجرای عددی | الگوریتم داخل evaluate همان شبکه است؛ predict و _predict وجود ندارند. forward استاندارد Model محورهای اضافه را مدیریت می‌کند. |
| اتصال داده | نگاشت inputs/outputs به Quantity، تبدیل layout و reshape مؤلفه‌ها حفظ شده‌اند. prepare_inputs و finish_outputs ابزار layout/packing هستند، نه dispatch الگوریتم شبکه. |
| اعتبارسنجی | شرط‌های وابسته به مقدار با check داخل evaluate اجرا می‌شوند؛ بررسی شکل و dtype با شرط معمولی انجام می‌شود. FPS مستقل PointNet++ نیز ماسک تهی را با check رد می‌کند. |
| رابط قدیمی | ModelInput/Output/Capabilities و representation adapterها برای داده/مرجع مانده‌اند؛ شبکه آن‌ها را ورودی نمی‌گیرد و خروجی نمی‌دهد. کد auxiliary/metadata حفظ شده ولی خروجی قابل انتخاب موتور نیست. |
| normalization | BatchNorm مشترک Tensor/list/mask و گزینهٔ instance در PointNet اضافه شده‌اند؛ آمار mask-aware نسبت به منبع تغییر رفتار دارد. در instance بعد از pooling normalization نداریم. |
| batch | PointNet و PointNet++ دارای BatchNorm فقط Tensor می‌پذیرند؛ حتی لیست هم‌اندازه و eval خطا دارند. padding خودکار نداریم. مدل‌های مجاز بدون BN لیست را map می‌کنند و گراف‌ها بدون padding به هم متصل می‌شوند. |
| GINO | ماسک با طول ثابت برای vmap و checkpoint کل فراخوانی بیرون vmap حفظ شده‌اند؛ مسیر predict/checkpoint داخلی جداگانه حذف شده است. |
| تغییرهای عددی قبلی | FPS با list+stack، endpoint در DeepNURBS با where، و ماسک/normalization در PointNetها حفظ شده‌اند؛ این‌ها فراتر از تغییر نام رابط‌اند. |

۲۸۳ تست جاری گذشتند؛ شامل قرارداد مستقیم Model، batch/list، خطای مقدار زیر vmap، گرادیان و checkpoint مشتق‌های بالاتر. خروجی‌ها و گرادیان پارامترهای هر ۱۱ شبکه نیز با snapshot پیش از این بازنویسی و وزن یکسان مقایسه و برابر بودند. این مقایسه روی تنظیمات نمونهٔ CPU است؛ تست‌های API حذف‌شدهٔ predict قرارداد فعال نیستند.

[گزارش تفصیلی و diff تاریخی](neural3dbench_comparison.fa.md) مربوط به snapshot پیش از حذف BaseNetwork است؛ برای رابط جاری از [راهنمای شبکه‌ها](networks.fa.md) استفاده شود.
