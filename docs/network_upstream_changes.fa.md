# تفاوت با شبکه‌های اصلی همکار

گزارش کاملِ فایل‌به‌فایل، تغییرات رفتار و سازگاری، و diff منبع در [مقایسهٔ تفصیلی با Neural3DBench](neural3dbench_comparison.fa.md) آمده است.

مبنای منبع، commit `1b635f4d1c3b47fab5c9e9a46b1824b789e478ee` از neural3dbench است؛ snapshot متناظر داخل تاریخچهٔ موتور در `80e5c8d` ثبت شده است. کدهای فعلی در `src/PINNCore/networks` هستند.

توضیح دقیق و قابل ارسال به همکار:

> معماری‌ها و هستهٔ محاسبات شبکه‌ها از کد تو آمده‌اند. تغییرات ما عمدتاً قرارداد ورودی/خروجی Model و batching موتور است که در فایل‌های مشترک قرار گرفته‌اند. برای normalization نیز لایه‌های مستقل Tensor/list و mask اضافه کردیم و شبکه‌ها را با import/factory و scope ماسک به آن متصل کردیم؛ encoder و decoderها را دوباره ننوشته‌ایم. پشتیبانی لیست در موتور و normalization ماسک‌پذیر قابلیت جدید ماست و در نسخهٔ مبنا وجود نداشت. PointNet و PointNet++ دارای BatchNorm در مسیر Model فقط Tensor می‌پذیرند؛ padding یا تبدیل خودکار لیست به batch انجام نمی‌دهیم.

## مرز تغییرات

| قسمت | تغییر واقعی |
|---|---|
| معماری شبکه‌ها | تعداد و ترتیب لایه‌ها، T-Net، message passing، attention، Fourier blocks، encoder/decoder اصلی حفظ شده‌اند. |
| رابط | BaseNetwork یک Model موتور شده؛ constructor نگاشت inputs/outputs می‌گیرد و forward عددی سابق با نام _predict اجرا می‌شود. predict(ModelInput) خروجی‌های قبلی و auxiliary/metadata را نگه می‌دارد. |
| محور و shape | تبدیل channel-first، reshape مؤلفه‌ها، vmap، استقلال، packing گراف و بازیابی لیست در _contract و helperهای موتور است. |
| normalization جدید | منطق Tensor/list/mask در PINNCore/networks/normalization.py است. در FNO، U-Net و GCN/GAT فقط کلاس BatchNorm از این ماژول import شده؛ forward این بلوک‌ها تغییر نکرده است. |
| PointNet | factory نرمال‌سازی، گزینهٔ instance، scope ماسک و پاک‌کردن ورودی نامعتبر اضافه شده‌اند. بدنهٔ forward خود encoder، T-Net و دو head دست نخورده است. normalization بعد از pooling در حالت instance طبق توافق حذف است. |
| PointNet++ | BatchNorm مشترک و scope ماسک متناظر با سطح grouping/interpolation اضافه شده است. الگوریتم اصلی نمونه‌برداری و propagation بازنویسی نشده است. |
| قرارداد ورودی PointNetهای دارای BatchNorm | در مسیر Model و Solver، هر ورودی لیستی رد می‌شود؛ این شامل لیست هم‌اندازه و حالت eval نیز هست. adapter قبلیِ padding خودکار حذف شده است. Tensor با ماسک صریح همچنان پشتیبانی می‌شود؛ لیست در خود لایهٔ مستقل BatchNorm و در مدل‌های بدون BatchNorm حفظ شده است. |
| تغییرات قبلی برای vmap | در FPS، بافر index به list+stack تبدیل شد؛ شاخهٔ شرطی endpoint در DeepNURBS به where تبدیل شد؛ در GINO، ماسک با طول ثابت جای boolean slicing نشست و جای checkpoint در مسیر Model تغییر کرد. این‌ها تغییرات واقعی‌اند و نباید حذفشان از توضیح را به معنی دست‌نخوردن کامل منبع گرفت. |

## شواهد بررسی

در این تغییر، AST بدنهٔ ۱۴ متد با snapshot مبنا مقایسه و برابر بود: PointNetEncoder.forward، TransformNet.forward، دو head مربوط به PointNet، GCNConv/GCNBlock/GATConv/GATBlock.forward، MeshGraphNetBlock.forward، FourierOperatorBlock3d/ProjectionHead3d.forward و ConvBlock3D/EncoderBlock3D/DecoderBlock3D.forward.

تست‌های dense، BatchNormهای جدید را با nn.BatchNorm اصلی جایگزین می‌کنند و خروجی، latent، کلیدهای state_dict و running buffers را مقایسه می‌کنند. تست ماسک صریح نیز دادهٔ واقعی یکسان را در Tensorهای dense با و بدون slot نامعتبر از PointNet و PointNet++ می‌گذراند و خروجی، گرادیان و بافرها را مقایسه می‌کند. رد ورودی لیستیِ PointNetهای دارای BatchNorm نیز بررسی می‌شود؛ تست آموزش خودکار روی لیست ragged بخشی از قرارداد فعلی نیست.

این بررسی اثبات یکسان‌بودن همهٔ فایل‌ها یا رفتار تمام configها نیست؛ قابلیت instance، آمار mask-aware و batching موتور عمداً رفتارهای جدیدند. هیچ کدی از auxiliary/metadata حذف نشده است.
