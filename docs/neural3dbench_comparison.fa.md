# بررسی تفصیلی تفاوت Neural3DBench اصلی و نسخهٔ Modelشدهٔ موتور

> یادداشت تغییر نام: پس از snapshot این گزارش، نام پکیج از `pinn_engine` به `PINNCore` و مسیر فعال به `src/PINNCore` تغییر کرد. نام‌های قبلی در متن و diff زیر برای حفظ دقت مقایسهٔ تاریخی نگه داشته شده‌اند. API جاری با `from PINNCore import ...` استفاده می‌شود.
> به‌روزرسانی رابط: پس از این snapshot، BaseNetwork و BaseGraphNetwork و مسیر predict حذف شدند؛ هر شبکه مستقیماً Model است و evaluate خودش را دارد. diff تاریخی زیر حفظ شده است؛ قرارداد جاری در [راهنمای شبکه‌ها](networks.fa.md) آمده است.

تاریخ بررسی: ۲۰۲۶-۱۱-۰۱. این گزارش وضعیت کد همراه همین فایل را توضیح می‌دهد. مبنای موتور پیش از تغییرات این نوبت، commit `2aa4ee5` است؛ تغییر قرارداد خروجی GeoCore و حذف padding خودکار PointNetها نیز در گزارش لحاظ شده‌اند.

## نتیجهٔ کوتاه برای همکار

> معماری‌ها، لایه‌های اصلی و الگوریتم‌های این شبکه‌ها از Neural3DBench آمده‌اند. ما آن‌ها را در پکیج موتور قرار داده‌ایم و قرارداد Quantity/Value، محورهای نام‌دار، batching خودکار و اتصال به Solver را اضافه کرده‌ایم. با این حال، تغییرات صرفاً تغییر نام ورودی/خروجی نیستند: API عمومی، روش batching، normalization ورودی ماسک‌دار، حالت instance جدید PointNet و اجرای ماسک/checkpoint در GINO تغییر واقعی دارند. معماری پایه حفظ شده، ولی یکسان‌بودن همهٔ رفتارهای اجرایی ادعا نمی‌شود.

## ۱. منبع، دامنه و روش مقایسه

- مخزن اصلی: [n94kholdi/neural3dbench](https://github.com/n94kholdi/neural3dbench).
- `HEAD` و `refs/heads/main` مخزن اصلی در این نوبت دوباره با `git ls-remote` بررسی شدند؛ هر دو همچنان `1b635f4d1c3b47fab5c9e9a46b1824b789e478ee` بودند. بنابراین commit جدیدی روی main نسبت به snapshot واردشده وجود نداشت. این بررسی ادعایی دربارهٔ branchهای دیگر ندارد.
- snapshot منبع در تاریخچهٔ موتور: `80e5c8d:third_party/neural3dbench/models`.
- محل کد فعال فعلی: `src/pinn_engine/networks`؛ import عمومی: `pinn_engine.networks`.
- تمام ۴۳ فایل Python در پوشهٔ models منبع با مسیر متناظر فعلی مقایسه شدند: ۱۴ فایل بدون تغییر متنی و ۲۹ فایل تغییرکرده. تفاوت صرف CRLF/LF در این شمارش تغییر محتوایی نیست.
- سه فایل منبع جدید نسبت به upstream وجود داشتند: `_contract.py`، `normalization.py` و adapter موقت `_point_batches.py`. آخری در قرارداد فعلی حذف شده است؛ دو فایل اول باقی‌اند.
- ۹۵ فایل upstream خارج از models در third_party حفظ شده‌اند؛ در مقایسهٔ این نوبت، اختلاف محتوایی نداشتند: ۱۶ فایل بایت‌به‌بایت یکسان و ۷۹ فایل فقط دارای تفاوت پایان خط بودند. این فایل‌ها عمدتاً مثال‌ها، تست‌ها، ابزارهای benchmark، داده‌ها و مستندات مرجعند؛ همگی به پکیج فعال موتور تبدیل نشده‌اند.
- `models/ModelInput.md` به `docs/network_input.md` منتقل شده و متن آن بدون تغییر مانده است. فایل LICENSE نیز همراه شبکه‌ها نگهداری و در بسته‌بندی لحاظ شده است.

در ادامه، «دلیل تغییر ما» توضیح تصمیمی است که در موتور گرفته‌ایم؛ دربارهٔ نیت همکار در انتخاب ساختار قبلی حدس نمی‌زنیم. اگر تغییری به تصمیم او مربوط می‌شود، آن را به‌عنوان موضوع قابل بازبینی مشخص کرده‌ایم.

## ۲. جابه‌جایی پکیج و سازگاری نصب

در upstream شبکه‌ها در یک پکیج عمومی به نام `models` بودند؛ اکنون زیر `pinn_engine.networks` نصب می‌شوند. importهای داخلی مانند `from models.base import BaseNetwork` به import نسبی در همان درخت تبدیل شده‌اند. ثبت شبکه‌ها در registry و exportهای اصلی حذف نشده‌اند؛ importهای registration در __init__ها متناسب با مسیر جدید اصلاح شده‌اند.

دلیل این تغییر، نصب شبکه‌ها همراه موتور بدون اضافه‌کردن third_party به sys.path و بدون تصاحب نام عمومی models بود. این تصمیم روی نام ماژول Python اثر دارد، نه بر فرمول convolution، attention یا message passing.

پیامدهای سازگاری:

1. کد مصرف‌کننده باید importهای models را به pinn_engine.networks تغییر دهد.
2. کلیدهای state_dict برای پیکربندی‌های قبلی حفظ شده‌اند؛ شبکه زیر یک فیلد wrapper جدید قرار نگرفته که پیشوند تازه‌ای به تمام وزن‌ها اضافه کند. داخل Solver، پیشوند ثبت خود Solver مانند `part_0.` اضافه می‌شود.
3. حفظ state_dict به معنی حفظ pickle کامل یک nn.Module نیست. مسیر کلاس‌ها و کلاس پایه عوض شده‌اند؛ بارگذاری شیء کامل ذخیره‌شده با مسیر قدیمی models.* تضمین نشده است.
4. فایل‌های قدیمی third_party، از جمله مثال‌ها و pyproject اصلی، مرجع upstream هستند. نصب موتور فقط runtime زیر src را ارائه می‌کند. مثال قدیمی با constructor قبلی و فراخوانی `network(ModelInput(...))` خودبه‌خود با API جدید سازگار نشده است.
5. شبکه‌ها همچنان nn.Module هستند؛ parameters، submodules، train/eval و انتقال device از سازوکار معمول PyTorch استفاده می‌کنند. هیچ wrapper صاحب وزنِ جداگانه اضافه نشده است.

## ۳. تغییر قرارداد عمومی BaseNetwork و registry

### ۳.۱. قبل و بعد

API upstream به‌صورت زیر بود:

```python
network = PointNet(config)
result = network(ModelInput(coordinates=points))  # ModelOutput
```

API موتور اکنون چنین است:

```python
xyz = Quantity("wing_points", (3,))
temperature = Quantity("temperature")
network = PointNet(
    config,
    inputs={"coordinates": xyz},
    outputs={"predictions": temperature},
)
result = network({xyz: Value(points, ("scenario", "point"))})
numeric_result = network.predict(ModelInput(coordinates=points))
```

در مثال دوم باید config با سه مؤلفهٔ مختصات و یک مؤلفهٔ خروجی سازگار باشد. نقش سمت چپ نگاشت، نام شناخته‌شدهٔ شبکه است؛ نام و هویت Quantity سمت راست را کاربر تعیین می‌کند.

### ۳.۲. تغییر دقیق متدها

| عضو | upstream | موتور فعلی | پیامد |
|---|---|---|---|
| BaseNetwork | subclass از nn.Module و ABC | subclass از Model و ABC؛ Model خود nn.Module است | وارد قرارداد محور، مقدار و Solver می‌شود. |
| __init__ | config اختیاری | config اختیاری، inputs و outputs اجباری | حتی استفادهٔ صرف از predict نیز این نگاشت‌ها را هنگام ساخت می‌خواهد. |
| forward | ModelInput می‌گرفت و ModelOutput می‌داد | نگاشت Quantity به Value می‌گیرد و نگاشت Value می‌دهد | API فراخوانی مستقیم قبلی شکسته شده است. |
| _predict | وجود نداشت | بدنهٔ عددی forward قبلی به این متد منتقل شده | امکان استفاده از همان الگوریتم در دو رابط فراهم شده است. |
| predict | self(inputs) را صدا می‌زد | validate_inputs و سپس _predict را مستقیم صدا می‌زند | خروجی عددی/ساختاری حفظ می‌شود؛ رفتار hook سطح خود شبکه فرق دارد. |
| signature | وجود نداشت | property مشتق‌شده از قرارداد شبکه و ماژول‌های واقعی | ورودی، خروجی، محورهای core و استقلال را اعلام می‌کند. |
| evaluate | وجود نداشت | پل بین مقادیر موتور و _predict | مسئول تبدیل layout و batching اختصاصی شبکه‌هاست. |

اعتبارسنجی نوع ModelInput، ورودی‌های اجباری و representation قبلی حفظ شده است. اعتبارسنجی‌های وابسته به داده در مسیر Model پیش از ورود به vmap انجام می‌شوند تا شرط‌هایی مانند torch.any روی BatchedTensor اجرا نشوند.

### ۳.۳. hook، compile و subclassهای مصرف‌کننده

`network(values)` از __call__ معمول nn.Module می‌گذرد و hook سطح شبکه را حفظ می‌کند. اما `network.predict(numeric_input)` اکنون یک فراخوانی مستقیم متد است؛ hookهای __call__/forward سطح خود شبکه و wrapper مربوط به forward کامپایل‌شده از این مسیر عبور نمی‌کنند. زیرماژول‌هایی که داخل _predict با فراخوانی معمول اجرا می‌شوند، hookهای خودشان را دارند. پس predict از نظر این رفتار، نسخهٔ کاملاً سازگارِ API قبلی نیست.

subclass بیرونی که فقط forward قبلی را پیاده کرده باشد نیز نیازمند تغییر است: باید _predict و قرارداد اتصال مناسب داشته باشد. هدف این تبدیل، native شدن شبکه‌ها در موتور بود؛ هزینه‌اش وابستگی مستقیم‌تر BaseNetwork به قرارداد موتور است. این تصمیم ساختاری باید با همکار بررسی شود اگر حفظ استفادهٔ مستقل Neural3DBench یک نیاز اصلی اوست.

### ۳.۴. registry

در ModelRegistry.create، inputs و outputs به پارامترهای keyword-only اجباری تبدیل شده‌اند و هنگام ساخت به کلاس شبکه منتقل می‌شوند. منطق ثبت نام، کشف config، خطای نام ناشناخته و تابع کمکی create_model حفظ شده است. create_model همچنان kwargs را به registry می‌رساند؛ بنابراین اکنون نگاشت‌ها از همان مسیر قابل ارسال‌اند.

ثبت یک subclass تازه به‌تنهایی کافی نیست: فایل _contract قراردادها را بر اساس config.name و دسته‌های مشخص شبکه می‌شناسد. برای معماری تازه باید مشخصات محور/نقش و احتمالاً مسیر اجرای اختصاصی را نیز توسعه داد. این یک محدودیت ساختاری افزوده نسبت به registry عمومی قبلی است؛ تحلیل خودکار قرارداد از بدنهٔ شبکه انجام نمی‌شود.

## ۴. Quantity، Value، shape و نقش‌های اختیاری

Quantity معرفی کمیت و شکل ذاتی آن است؛ Value مقدار Tensor یا لیست Tensor به همراه نام محورهای نمونه است. مثلاً Quantity برداری با شکل ذاتی (3,) می‌تواند Value با محورهای scenario و point و Tensor به شکل (S,P,3) داشته باشد. نام محور با اندازهٔ آن یکی نیست؛ دو محور هم‌اندازه فقط به علت برابر بودن عددی به هم match نمی‌شوند.

initialize_contract موارد زیر را هنگام ساخت بررسی می‌کند:

- inputs و outputs باید نگاشت نقش به Quantity باشند؛ نقش ناشناخته پذیرفته نیست.
- ورودی‌های واقعاً لازم طبق config باید وجود داشته باشند؛ مثلاً feature_dim/input_dim غیرصفر، point_features را اجباری می‌کند.
- حداقل یک خروجی عددی باید انتخاب شود.
- هر نقش در یک سمت Quantity متمایز دارد؛ ورودی و خروجی نمی‌توانند همان Quantity باشند.
- حاصل‌ضرب شکل ذاتی Quantity باید با تعداد مؤلفهٔ آن نقش برابر باشد. نام یکسان با شکل‌های ناسازگار نیز رد می‌شود.

این قرارداد عمداً اجازه می‌دهد ۹ کانال خروجی شبکه به Quantity با شکل (3,3) تبدیل شود، یا یک کانال به Quantity اسکالر با شکل (). تبدیل صرفاً reshape مؤلفه‌هاست؛ محور point یا scenario در شکل ذاتی ادغام نمی‌شود. از طرف دیگر این قرارداد یک مدل in-place با Quantity ورودی و خروجی یکسان را در این bindingها نمی‌پذیرد.

ورودی اختیاری مانند point_mask یا query_coordinates فقط با حضور در inputs هنگام init انتخاب می‌شود؛ پس از انتخاب، هنگام اجرا الزامی است. تغییر انتخاب نقش‌ها بین فراخوانی‌ها بخشی از این API نیست. عدم انتخاب آن در init یعنی مقدار None در ModelInput عددی تولیدشده برای آن نقش.

تمام فیلدهای ModelInput منبع، نقش موتور نیستند. فیلدهایی مانند physical_parameters، time، boundary_conditions، material_properties، batch و metadata در dataclass اصلی مانده‌اند، ولی binding فعلی تنها نقش‌هایی را ارائه می‌کند که شبکه‌های موجود و _specs پشتیبانی می‌کنند. نگهداری dataclass عمومی به معنی مصرف تمام این فیلدها توسط هر معماری نیست.

### خروجی‌های انتخاب‌شده و کار اضافه

در مسیر Value فقط predictions و latent عددی قابل انتخاب‌اند؛ GCN latent ندارد. auxiliary و metadata در predict باقی مانده‌اند و کد تولیدشان حذف نشده است. انتخاب خروجی تنها مشخص می‌کند چه مقداری به موتور تحویل داده شود؛ _predict لزوماً headها یا auxiliaryهای انتخاب‌نشده را کنار نمی‌گذارد. مثلاً درخواست فقط latent تضمین نمی‌کند head پیش‌بینی اجرا نشود. pruning داخل خود شبکه در این تبدیل پیاده نشده است.

## ۵. قرارداد محور و layout هر شبکه

در جدول زیر، شکل‌ها مربوط به یک هندسه‌اند؛ S نمایندهٔ محورهای اضافی batch است که ممکن است بیش از یک محور باشند. C/O/F تعداد مؤلفه و جزو شکل ذاتی Quantity است.

| شبکه | نقش‌های ورودی موتور و محور core | predictions | latent |
|---|---|---|---|
| GCN | node_features: vertex؛ edge_index: edge | vertex با O مؤلفه | ندارد |
| GAT | node_features: vertex؛ edge_index: edge | vertex با O مؤلفه | vertex با hidden_dim مؤلفه |
| MeshGraphNet | node_features: vertex؛ edge_index/edge_features: edge | vertex با O مؤلفه | vertex با latent_dim مؤلفه |
| PointNet | coordinates و features/mask انتخاب‌شده: point | point در segmentation؛ بدون point در classification | embedding بدون point با global_dim مؤلفه |
| PointNet++ | همان قرارداد PointNet | همان دو حالت | embedding بدون point با global_dim مؤلفه |
| SIREN | coordinates و features/mask انتخاب‌شده: point | point | point با hidden_dim مؤلفه |
| Transolver | coordinates و features/mask انتخاب‌شده: point | point | point با hidden_dim مؤلفه |
| GINO | coordinates/features/mask: point؛ query_coordinates اختیاری: query | با query جدا محور query؛ در غیر این صورت point | latent_z, latent_y, latent_x با width مؤلفه |
| FNO3D | voxel_fields: z,y,x | z,y,x | همان محورهای grid با width مؤلفه |
| UNet3D | voxel_fields: z,y,x | z,y,x | bottleneck_z,bottleneck_y,bottleneck_x با کانال bottleneck |
| DeepNURBS | coordinates دوبعدی در فضای پارامتری: point | point با output_dim مؤلفه | میدان عصبی پیش از اعمال قید، همان محور و مؤلفه |

تبدیل‌های عددی داخل bridge:

1. دادهٔ voxel موتور channel-last است؛ (S,Z,Y,X,C) پیش از الگوریتم اصلی به (B,C,Z,Y,X) تبدیل می‌شود و خروجی برعکس برمی‌گردد.
2. latent سه‌بعدی GINO نیز از channel-first به channel-last تبدیل می‌شود.
3. edge_index موتور به شکل (E,2) با dtype صحیح است؛ برای الگوریتم اصلی به (2,E) و long تبدیل می‌شود. offset گراف‌ها پیش از تبدیل اعمال می‌شود.
4. point_mask موتور یک Quantity تک‌مؤلفه است؛ بعد مؤلفهٔ موقت آن هنگام ساخت ModelInput حذف می‌شود تا (B,P) شود.
5. برای یک فراخوانی core بدون محور batch، بعد B=1 به الگوریتم اصلی داده و هنگام بازگرداندن خروجی حذف می‌شود.
6. latent کوچک‌شدهٔ UNet نام محور مستقل گرفته است تا با z/y/x خروجی کامل اشتباهاً هم‌اندازه فرض نشود.

## ۶. batching، لیست و استقلال

### ۶.۱. مدل‌هایی که batch را به موتور می‌سپارند

در نبود BatchNorm و مسیر گراف، signature فقط محورهای core را می‌خواهد. Model.forward محورهای dense اضافه را با torch.func.vmap اعمال می‌کند. ورودی فاقد یک محور batch در آن محور shared است؛ ورودی‌های دارای همان نام محور به‌صورت جفت‌شده استفاده می‌شوند. برای محور لیستیِ اضافی، اعضا با حلقهٔ ترتیبی اجرا می‌شوند و نتیجه لیست می‌ماند؛ job/thread یا اجرای موازی CPU افزوده نشده است.

BaseNetwork برای dropout مقدار vmap_randomness="different" دارد تا اعضای batch mapped نمونه‌های تصادفی مستقل بگیرند. این انتخاب حفظ توزیع مدنظر dropout است؛ ترتیب بیت‌به‌بیت اعداد تصادفی در مقایسه با forward batched upstream تضمین نمی‌شود.

### ۶.۲. مدل‌هایی که کل batch را می‌گیرند

تمام شبکه‌های گرافی و هر شبکه‌ای که واقعاً یک زیرماژول BatchNorm ثبت‌شده داشته باشد، در signature محور ... دارند و Values کامل را به evaluate می‌گیرند. تشخیص BatchNorm از خود modules انجام می‌شود، نه صرفاً نام گزینهٔ config. بنابراین داشتن یک BatchNorm ثبت‌شده ولی استفاده‌نشده هم این مسیر محافظه‌کارانه را فعال می‌کند.

_prepared_batches محورهای batch را با نام پیدا می‌کند، محورهای غایب را broadcast می‌کند، محورهای dense اضافه را در B تخت می‌کند و برای لیست، گروه‌های جدا نگه می‌دارد. ترتیب محور batch از نگاشت inputs هنگام تعریف گرفته می‌شود؛ ترتیب اتفاقی دیکشنری values تعیین‌کننده نیست. این تابع padding انجام نمی‌دهد.

### ۶.۳. قرارداد فعلی PointNet و PointNet++ دارای BatchNorm

در مسیر Model/Solver، اگر یکی از این دو شبکه BatchNorm داشته باشد، **هر ورودی Value لیستی رد می‌شود**؛ حتی لیست با اعضای هم‌اندازه و حتی در eval با running statistics ثابت. ورودی باید Tensor باشد. Tensor دارای point_mask صریح همچنان پذیرفته است.

adapter موقت _point_batches.py که لیست را به Tensor دارای padding تبدیل می‌کرد حذف شده است. موتور دیگر برای این شبکه‌ها padding یا تبدیل خودکار لیست به batch اضافه نمی‌کند. کاربری که دادهٔ هم‌اندازه دارد Tensor می‌دهد؛ کاربری که Tensor از قبل padded می‌دهد مسئول انتخاب نمایش است و mask آن در normalization و pooling رعایت می‌شود. list در لایهٔ مستقل BatchNorm و list در مدل‌های بدون BatchNorm همچنان قابلیت جداگانه و معتبرند.

این تصمیم یک محدودیت عمدی API فعلی است، نه ناممکن‌بودن ریاضی normalization روی لیست. همچنین حذف adapter به معنی حذف paddingی نیست که خود GeoCore در استفادهٔ مستقیمِ point_cloud با تعداد نقاط متفاوت انجام می‌دهد؛ آن رفتار در کتابخانهٔ همکار از قبل وجود داشت.

### ۶.۴. gridهای با اندازهٔ متفاوت

FNO/UNet بدون BatchNorm می‌توانند اعضای لیست را جدا از مسیر معمول مدل پردازش کنند، مشروط به محدودیت‌های عددی خود شبکه. با BatchNorm و آمار batch، gridهای واقعاً متفاوت خطا می‌دهند؛ padding عمومی برای FFT یا convolution افزوده نشده است. در eval با آمار running ثابت، گروه‌های متفاوت می‌توانند جدا اجرا شوند. اگر اعضای لیست هم‌اندازه باشند، مسیر BatchNorm آن‌ها را یک batch مشترک در نظر می‌گیرد. محدودیت سخت‌تر «تمام لیست‌ها ممنوع» در بند قبل مخصوص PointNet و PointNet++ است.

### ۶.۵. قواعد استقلال و اثر بر مشتق

| شبکه/محور | استقلال اعلام‌شده |
|---|---|
| SIREN و DeepNURBS، point | True؛ هر نقطه جدا محاسبه می‌شود. |
| PointNet و PointNet++، point مشترک ورودی/خروجی | False؛ pooling، T-Net، normalization یا تجمیع نقاط می‌تواند coupling بسازد. |
| Transolver، point | False؛ attention بین نقاط ارتباط ایجاد می‌کند. |
| FNO/UNet، محورهای مکانی مشترک | False؛ عملیات مکانی چند نقطه را به هم مرتبط می‌کند. |
| GCN/GAT/MeshGraphNet، vertex | False؛ message passing وابستگی بین رأس‌ها می‌سازد. |
| GINO با query جدا، query | True نسبت به خود queryها؛ تغییر یک query، source encoder یا query دیگر را عوض نمی‌کند. |
| GINO با query برابر source، point | False؛ همان مختصات در encoding منبع نیز دخالت دارند. |
| ... در مدل‌های whole-batch | در صورت استفاده از آمار batch، False؛ با running statistics ثابت، True. |

این جدول قرارداد دستی متناسب با معماری‌های موجود است؛ dataflow را از بدنهٔ arbitrary شبکه اثبات نمی‌کند. تغییر یک شبکه، مثلاً افزودن لایهٔ وابسته به چند نقطه به SIREN، مستلزم بازبینی signature است. صرف اسکن BatchNorm، استقلال همهٔ محورهای دیگر را اثبات نمی‌کند.

backend AD موتور مسیر مشتقِ دارای محور مشترکِ dependent را با خطای صریح رد می‌کند؛ backward معمول آموزش شبکه‌ها برقرار است. برای محورهای وابسته، در این تغییر Jacobian کامل/بلوک‌قطری یا مشتق اختصاصی spectral/mesh پیاده نشده است. rename کردن محورها نیز جای تحلیل وابستگی واقعی را نمی‌گیرد. دامنهٔ هندسی Quantityها و بررسی سازگاری domain ورودی/خروجی هنوز کار آینده است.

## ۷. packing شبکه‌های گرافی

upstream الگوریتم graph را روی node_features به شکل (N,C) و edge_index به شکل (2,E) اجرا می‌کرد. موتور برای چند گراف، nodeها و edge featureها را concatenate می‌کند و به index یال‌های هر گراف offset تعداد رأس‌های قبلی را می‌افزاید. الگوریتم اصلی یک بار اجرا می‌شود و خروجی با طول‌های ثبت‌شده split می‌شود. بین گراف‌ها یال جدید ساخته نمی‌شود؛ شکل عمومی خروجی لیست، لیست باقی می‌ماند.

دلیل تصمیم ما، اجرای یک graph disjoint با همان message-passing موجود و پرهیز از padding رأس/یال بود. حلقه‌ها، cat/stack و bookkeeping در bridge هستند؛ convolution و attention گراف دوباره نوشته نشده‌اند.

اما «نبود یال بین گراف‌ها» به‌تنهایی به معنی استقلال همهٔ محاسبات نیست. اگر GCN/GAT با BatchNorm آموزش داده شوند، آمار بر اجتماع تمام رأس‌های packed گرفته می‌شود. گراف ۴۰ رأسی دو برابر گراف ۲۰ رأسی در میانگین هر کانال سهم دارد. این با اجرای دو forward مستقل و آمار جدا فرق دارد. با running statistics ثابت در eval، آن coupling آماری وجود ندارد.

MeshGraphNet فعلی اصلاً گزینهٔ BatchNorm ندارد؛ normalization آن layer_norm یا none است. LayerNorm روی ویژگی‌های هر رأس/یال جدا اعمال می‌شود. بنابراین نباید نتیجهٔ BatchNorm بین‌گرافی GCN/GAT را به MeshGraphNet نسبت داد.

فیلد batch قدیمی ModelInput در binding موتور ارائه نشده و packing یک batch vector به _predict نمی‌فرستد؛ lengths و offsets داخلی‌اند. API عددی ModelInput همچنان آن فیلد را دارد و اطلاعات ساختاری/metadata قبلی نیز در همان مسیر موجودند. اگر معماری تازه از batch برای graph-wise pooling استفاده کند، bridge فعلی باید توسعه یابد.

## ۸. normalization مشترک: تغییر دقیق و معنا

### ۸.۱. کلاس‌های BatchNorm جدید

BatchNorm1d/2d/3d در networks/normalization.py subclass کلاس‌های متناظر PyTorch هستند. در حالت Tensor بدون mask، مستقیماً forward native را اجرا می‌کنند. constructor پارامترهای معمول PyTorch را نگه می‌دارد؛ eps، affine، momentum، running_mean/var و num_batches_tracked به پیاده‌سازی native سپرده می‌شوند.

تفاوت نوع واقعی کلاس وجود دارد: isinstance(layer, nn.BatchNorm1d) درست است، ولی type(layer) is nn.BatchNorm1d لزوماً درست نیست. جایگزینی در constructorهای FNO، UNet، GCN/GAT و PointNet++ از همین کلاس‌های مشترک است؛ فرمول forward بلوک‌های اصلی آن‌ها با این import عوض نشده است.

### ۸.۲. ورودی list و mask در لایهٔ مستقل

لیست در این utility به معنی قطعات یک batch مشترک است. اعضا rank، تعداد کانال، dtype و device مشترک دارند، ولی اندازهٔ B و محورهای مکانی می‌تواند متفاوت باشد. layout تابع channel-first یعنی (B,C,*spatial) است؛ این layout با Value عمومی channel-last موتور یکی نیست.

تابع هر عضو را به ردیف‌های نمونه با ستون C تبدیل می‌کند، فقط ردیف‌های معتبر mask را برمی‌دارد، ردیف‌های واقعی را به هم می‌چسباند و BatchNorm native را یک بار اجرا می‌کند. سپس خروجی split و به شکل هر عضو برگردانده می‌شود. در جایگاه‌های mask=False صفر قرار می‌گیرد. در نتیجه:

- هر نمونهٔ معتبر وزن یکسان دارد؛ هر هندسه الزاماً وزن یکسان ندارد.
- یک فراخوانی روی لیست، running statistics را یک بار به‌روزرسانی می‌کند.
- padding و NaNهای نامعتبر وارد آمار نمی‌شوند؛ گرادیان ورودی نامعتبر صفر است.
- شرط حداقل تعداد نمونهٔ native BatchNorm در حالتی که آمار batch لازم است برقرار می‌ماند.
- ماسک باید bool با شکل (B,*spatial) و هم‌device داده باشد؛ به اندازهٔ کانال محور ندارد.
- dynamic indexing، cat و scatter هزینه دارند؛ از وجود این utility تضمین fullgraph compilation یا افزایش سرعت حاصل نمی‌شود.

این قابلیت با ممنوعیت list در ورودی Modelِ PointNet دارای BatchNorm تناقض ندارد: utility مستقل یک لایهٔ عددی است؛ اجرای همهٔ لایه‌های یک شبکه روی list، تصمیم جداگانه‌ای می‌خواهد و در قرارداد فعلی خودکار انجام نمی‌شود.

### ۸.۳. انتقال mask با context

normalization_mask از ContextVar برای scope ماسک همان فراخوانی استفاده می‌کند؛ امضای تک‌تک Conv/MLPها عوض نشده است. پس از خروج، حتی exception، context قبلی برگردانده می‌شود. این state محلی فراخوانی است؛ cache دائمی batch یا hook/monkey patch روی ماژول نیست.

لایهٔ فضایی mask متناظر خودش را می‌گیرد. لایهٔ pooled با شکل (B,C) mask فضایی را به ارث نمی‌برد؛ اگر BatchNorm دارد، آمار روی B embedding واقعی گرفته می‌شود. این تمایز برای T-Net و head classification مهم است.

masked_point_inputs پیش از convolution، مختصات و feature نامعتبر را صفر می‌کند. فقط ماسک‌کردن خروجی BN کافی نبود؛ NaN در ورودی convolution می‌توانست گرادیان وزن‌ها را آلوده کند. این یک اصلاح رفتار ورودی ماسک‌دار است، نه تغییر صرف interface.

در upstream، mask در pooling/خروجی رعایت می‌شد اما آمار برخی normalizationها هنوز ردیف‌های نامعتبر را می‌دید. اکنون آمار فضایی هم mask را رعایت می‌کند؛ بنابراین برای ورودی صریحاً ماسک‌دار، هم‌ارزی کامل عددی با رفتار قدیمی ادعا نمی‌شود.

### ۸.۴. InstanceNorm1d و حالت جدید PointNet

InstanceNorm1d جدید برای هر نمونه و هر کانال جدا روی point آمار می‌گیرد. بدون mask از مسیر native می‌گذرد؛ روی list هر عضو را مستقل اجرا می‌کند. در حالت ماسک‌دار، track_running_stats باید False باشد؛ mean/variance فقط روی نقاط معتبر است، half/bfloat در float32 accumulate می‌شود، affine اعمال می‌شود و slot نامعتبر صفر می‌ماند. حالت تماماً نامعتبر در خود utility صفر می‌شود، ولی ورودی عمومی PointNet همچنان هندسهٔ بدون نقطهٔ معتبر را رد می‌کند.

PointNet اکنون سه گزینه دارد:

| گزینه | پیش از pooling | بعد از pooling |
|---|---|---|
| batch، پیش‌فرض اصلی | BatchNorm مشترک روی B و point برای هر کانال | BatchNorm روی B embedding |
| none، گزینهٔ قبلی | normalization ندارد | normalization ندارد |
| instance، گزینهٔ جدید | InstanceNorm روی point هر هندسه، affine=True، بدون running stats | normalization ندارد؛ bias لایه‌های FC بدون normalization فعال است. |

این سیاست instance طبق توافق این پروژه اضافه شد؛ به منبع upstream نسبت داده نمی‌شود. در حالت جدید، حضور biasها و running bufferها با batch فرق دارد؛ ادعای حفظ شمار/کلیدهای پارامترهای config قبلی را نباید به گزینهٔ جدید تعمیم داد. خود ساختار T-Net، max-pooling، encoder و headهای اصلی دوباره نوشته نشده است. PointNet++ فعلاً همان batch و none را دارد؛ instance به config آن افزوده نشده است.

### ۸.۵. normalization در سایر شبکه‌ها

FNO همچنان none/batch/instance و UNet همچنان batch/instance/group/none دارند؛ instance سه‌بعدی آن‌ها native باقی مانده است. GCN گزینهٔ instance_norm قبلی را با nn.InstanceNorm1d روی ورودی (N,C) حفظ کرده است؛ آن را به «نرمال‌سازی روی رأس‌های هر گراف» تبدیل نکرده‌ایم. تفسیر محور این گزینهٔ منبع باید جدا بررسی شود و ارتباطی با InstanceNorm نقطه‌ای جدید PointNet ندارد. LayerNormهای Transolver و MeshGraphNet نیز تغییر عددی داده نشده‌اند.

## ۹. تغییرات اختصاصی هر معماری

تغییر مشترک constructor، importها و انتقال forward به _predict برای هر ۱۱ شبکه اعمال شده است. بندهای زیر تغییرهای اضافه بر آن پایهٔ مشترک را توضیح می‌دهند.

### ۹.۱. PointNet

PointNetConfig فقط پذیرش normalization="instance" را به گزینه‌های قبلی اضافه کرده است. SharedPointMLP برای ساخت norm از factory مشترک استفاده می‌کند. در TransformNet و classification head، انتخاب norm بعد از pooling و bias FCها متناسب با گزینهٔ جدید شده است؛ در batch/none قبلی، انتخاب پارامترها حفظ شده است.

در _predict، پاک‌کردن ورودی نامعتبر و scope normalization_mask دور همان encoder/head موجود اضافه شده است. encoder، T-Net و دو head در بدنهٔ forward خود بازنویسی نشده‌اند؛ کاربرد آن‌ها از همان pipeline قبلی است. regularizer مربوط به feature transform و auxiliary transformها حذف نشده‌اند و از predict قابل دریافت‌اند.

در segmentation، خروجی نقاط نامعتبر همچنان صفر می‌شود. در classification، point کاهش یافته و برای هر سناریو بردار خروجی داریم؛ latent نیز embedding سراسری است. وابستگی خروجی به مجموعهٔ نقاط با «مستقل‌بودن scenarioها در نبود BN» فرق دارد.

### ۹.۲. PointNet++

در farthest_point_sample، الگوریتم انتخاب نقطهٔ دورتر از centroid و به‌روزرسانی کمترین فاصله حفظ شده است؛ تخصیص torch.empty و نوشتن inplace در ستون index به list.append و torch.stack تبدیل شده تا vmap بتواند نتیجهٔ mapped را تولید کند. این list داخلی برای جمع‌کردن indexهاست و با Value لیستیِ هندسه‌ها فرق دارد.

بررسی torch.any(valid_counts==0) از داخل helper حذف شده؛ ورودی عمومی شبکه این اعتبارسنجی را پیش از vmap انجام می‌دهد. helper همچنان export عمومی دارد، پس فراخوانی مستقیم آن با mask تهی دیگر همان خطای قبلی را تضمین نمی‌کند. این یک افت کوچک سازگاری در helper مستقل است و باید صریح ثبت شود.

SharedPointMLP2d از BatchNorm2d مشترک استفاده می‌کند. SetAbstraction ماسک normalization را با سطح grouping منطبق می‌کند: در group_all از ماسک منبع و در grouping محلی از معتبر بودن centroid استفاده می‌شود. slotهای اضافی radius grouping که upstream با نزدیک‌ترین همسایهٔ واقعی پر می‌کرد، همچنان در آمار محلی همان وزن تکرار را دارند؛ تنها centroid/slot بیرونی نامعتبر حذف می‌شود. پس «هر slot ماسک‌شده از تمام normalizationها حذف شده» توصیف دقیقی نیست.

FeaturePropagation هنگام اجرای SharedPointMLP ماسک نقاط مقصد را scope می‌کند. interpolation سه‌همسایه، max aggregation، انتخاب سطح‌ها و decoder segmentation اصلی حفظ شده‌اند. در _predict نیز ورودی نامعتبر پاک و scope اصلی mask اضافه شده است. API فعلی برای این شبکه دارای BN، list را رد می‌کند و adapter padding ندارد.

### ۹.۳. GINO: ماسک با طول ثابت

upstream sourceهای معتبر را با boolean slicing جدا می‌کرد و queryهای پیش‌فرض نامعتبر را اصلاً به decoder نمی‌داد. موتور برای سازگاری با vmap، طول Tensorها را ثابت نگه می‌دارد:

1. KernelIntegralTransform آرگومان source_mask می‌گیرد؛ شکل و dtype آن بررسی می‌شود.
2. source coordinate/feature نامعتبر پیش از projection/kernel صفر می‌شود تا NaN یا padding وارد محاسبهٔ معتبر نشود.
3. وزن kernel در source_mask ضرب می‌شود؛ سهم نقطهٔ نامعتبر در numerator و denominator صفر است.
4. default query نامعتبر پاک می‌شود، decoder برای طول کامل اجرا می‌شود و پیش‌بینی آن query در انتها صفر می‌شود.
5. حالت source یا target واقعاً با طول صفر خروجی صفر با شکل درست می‌دهد؛ guard مربوط به target خالی افزوده شده است.

فرمول compact-support kernel، source projection، MLP kernel، cutoff شعاعی و نرمال‌سازی weighted integral حفظ شده‌اند. اما مقدار کار ثابت نمانده است: slot نامعتبر هنوز ممکن است وارد projection، محاسبهٔ فاصله و kernel/decoder شود و در انتها سهم صفر بگیرد. source_count اکنون طول کامل را می‌بیند، بنابراین اندازهٔ chunkها نیز می‌تواند با حالت sliced فرق کند. ماسک پرتعداد ممکن است کار اضافهٔ قابل توجه بسازد. برابری ریاضی نقاط معتبر به معنی برابر بودن شمار عملیات یا ترتیب گردکردن نیست.

latent grid، Fourier blocks و output projection از معماری اصلی‌اند. latent_coordinates، query_is_source، point_mask و metadataهای قبلی در ModelOutput عددی باقی مانده‌اند. در Model فقط نقش‌های عددی انتخاب‌شده به Value تبدیل می‌شوند.

### ۹.۴. GINO: محل checkpoint

upstream داخل KernelIntegralTransform، query chunkهای integral را هنگام فعال بودن grad checkpoint می‌کرد. نگهداری این checkpoint داخل تابعی که موتور آن را vmap می‌کند با saved-tensor hookهای آن مسیر ناسازگار بود. در مسیر Model، checkpoint داخلی صریحاً خاموش و checkpoint بیرون تمام vmapها گذاشته شده است:

```python
checkpoint(mapped_function, *arguments, use_reentrant=False)
```

این رفتار با checkpoint_batches=True در GINO فعال است؛ پیش‌فرض Modelهای دیگر False است. در نبود grad، checkpoint انجام نمی‌شود. در batch لیستی، هر عضو از مسیر خودش checkpoint می‌شود؛ در نبود batch نیز فراخوانی core پوشش دارد. مسیر wildcard از این checkpoint خودکار عبور نمی‌کند؛ شبکهٔ GINO فعلی از مسیر معمول استفاده می‌کند.

تفاوت مهم: replay اکنون کل تابع mapped شامل source projection، encoder، Fourier blocks، decoder و projection خروجی را می‌تواند دربرگیرد، نه فقط integral یک chunk. این کار saved tensor کمتری نگه می‌دارد ولی محاسبات replay را گسترده‌تر می‌کند. محدودیت workspace زوج‌نقاط و chunkهای kernel همچنان وجود دارد. تضمین کاهش مشخص peak حافظه یا افزایش سرعت نداریم.

predict(ModelInput) در GINO override شده و checkpoint داخلی kernel را نگه می‌دارد؛ بنابراین دو API جای checkpoint یکسانی ندارند. فعال‌کردن همین سیاست برای مدل‌های stateful مانند BN بدون بررسی می‌تواند state را هنگام replay دوباره تغییر دهد؛ در این تبدیل عمومی فعال نشده است. خروجی و گرادیان‌های GINO تا مرتبهٔ سوم در تست‌های موجود با حالت بدون checkpoint مقایسه شده‌اند؛ از این تست پشتیبانی همهٔ ترکیب‌های torch.func/compile نتیجه نمی‌شود.

### ۹.۵. DeepNURBS

در _bspline_basis، شاخهٔ Python مبتنی بر torch.any(at_right_endpoint) حذف شده است. endpoint_basis اکنون همیشه ساخته می‌شود و torch.where انتخاب را انجام می‌دهد؛ این تغییر امکان ورودی mapped در endpoint راست را می‌دهد. مقدار basis در همان شرایط باید حفظ شود، ولی Tensor کوچک endpoint حتی وقتی هیچ endpointی وجود ندارد ساخته می‌شود.

geometry و NURBS definition همچنان در config/بافرهای شبکه‌اند؛ آن‌ها به Quantity ورودی متحرک تبدیل نشده‌اند. ورودی runtime همان UV دوبعدی است. اعمال admissible/lifting field و معنای latent به‌عنوان میدان عصبی قبل از قید حفظ شده است. فقط رابط، endpoint branch و batching تغییر کرده‌اند؛ مدل به یک encoder عمومی geometry تبدیل نشده است.

### ۹.۶. FNO3D

علاوه بر قرارداد عمومی، BatchNorm3d داخل FourierOperatorBlock3d به کلاس مشترک تغییر کرده است. spectral.py، پارامترهای طیفی، انتخاب modeها، coordinate grid، lifting، padding خود FNO، ترتیب Fourier/skip branch و projection اصلی حفظ شده‌اند. padding داخلی خود معماری FNO با adapter حذف‌شدهٔ padding لیست PointNet ارتباطی ندارد.

هیچ مشتق spatial اختصاصی FFT در این تغییر افزوده نشده است؛ مسیر differentiation موتور همان قرارداد AD است. Grid channel layout در bridge عوض می‌شود، نه در spectral convolution. whole-batch بودن فقط با وجود واقعی BatchNorm فعال می‌شود.

### ۹.۷. UNet3D

factory normalization فقط برای گزینهٔ batch، BatchNorm3d مشترک را می‌سازد؛ instance، group و none قبلی باقی‌اند. ConvBlock، encoder/decoder، skip connectionها و منطق upsample/pad/crop اصلی بازنویسی نشده‌اند. latent در binding نام محورهای bottleneck مستقل دارد، زیرا resolution آن با prediction برابر نیست. grid ragged با BN فعال هنوز اجرای عمومی مشترک ندارد.

### ۹.۸. GCN

غیر از import، constructor و نام متد عددی، الگوریتم GCN و آماده‌سازی یال/درجه/normalization گراف حفظ شده‌اند. BatchNorm1d بلوک از کلاس مشترک است؛ InstanceNorm1d قبلی اصلاح یا بازتعریف نشده است. شبکه در API اصلی latent جدا نمی‌داد و موتور نیز latent اختراع نکرده است. batching جدیدِ گراف disjoint در bridge اتفاق می‌افتد.

### ۹.۹. GAT

attention، چندhead، incoming softmax، self-loop و return-attentionهای عددی اصلی حفظ شده‌اند. BatchNorm بلوک مشترک شده است. latent hidden_dim با محور vertex به موتور ارائه می‌شود؛ auxiliaryهای attention فقط در predict باقی‌اند. packing و اثر آماری BatchNorm بین گراف‌ها تغییر اجرایی بیرون بلوک attention است.

### ۹.۱۰. MeshGraphNet

encoder، processor، decoder و blocks.py متن یکسان با snapshot دارند. edge-first interaction، جمع receiver-wise، residual updateهای node/edge و MLPهای اصلی حفظ شده‌اند. model.py فقط تغییرهای عمومی import/constructor/نام forward دارد. این شبکه همیشه کل batch گراف را برای packing می‌گیرد، حتی بدون BN. normalization همان layer_norm/none منبع است و decoder اصلی بدون norm می‌ماند.

ابزارهای adaptive meshing و rollout موجود در third_party به Model موتور تبدیل نشده‌اند. حفظ فایل آن‌ها به معنی برنامه‌ریزی remeshing یا state transfer توسط Solver نیست.

### ۹.۱۱. SIREN

siren/layers.py و SineLayer بدون تغییرند؛ frequencyها، initialization و محاسبهٔ شبکه عوض نشده‌اند. model.py تغییرهای عمومی import/constructor/انتقال forward را دارد. موتور point را در signature core نگه می‌دارد و محورهای اضافی را map می‌کند؛ استقلال point از ساختار نقطه‌ای فعلی می‌آید. مشتق اختصاصی یا backward دست‌نویس برای آن افزوده نشده است.

### ۹.۱۲. Transolver

block.py، components.py و physics_attention.py بدون تغییر متنی‌اند. embedding، placeholder، slicing attention، feed-forward و output head اصلی همان‌اند. model.py فقط تغییرهای عمومی اتصال را دارد. شبکه بدون BN است، پس batchهای اضافی با vmap اجرا می‌شوند؛ coupling روی point ناشی از attention صریحاً در signature ثبت شده است. return_attention_weights و auxiliary آن در predict موجودند، ولی binding خروجی ساختاری موتور هنوز ارائه نمی‌شود.

## ۱۰. فهرست فایل‌های جدید و مسئولیت تک‌تک helperها

### _contract.py

| تابع/ثابت | مسئولیت و تصمیم افزوده |
|---|---|
| _GRAPHS، _GRIDS، _SPATIAL | دسته‌بندی صریح نام معماری و نام محورهای grid؛ افزودن معماری جدید ممکن است به تغییر این سیاست‌ها نیاز داشته باشد. |
| _specs | نقش‌ها، مؤلفه‌ها، محورهای core و ورودی‌های required هر معماری؛ معنای شبکه از config.name انتخاب می‌شود. |
| initialize_contract | اعتبارسنجی نگاشت init، شکل ذاتی، نقش‌های اختیاری و متمایز بودن Quantityها. |
| contract_signature | افزودن ... برای graph/BN، انتخاب محور query و قواعد independence؛ تحلیل dataflow خودکار نیست. |
| _prepared_batches | broadcast نام‌دار، flatten محورهای dense، گروه‌بندی list، منع list در PointNetهای BN؛ بدون padding. |
| _model_input | تبدیل layout grid و mask به ModelInput عددی موجود. |
| validate_contract | بررسی Quantity/Value و اجرای validation عددی پیش از vmap؛ برای گراف هر عضو پیش از packing بررسی می‌شود. |
| _outputs | انتخاب نقش عددی، برگرداندن channel-last و reshape شکل ذاتی Quantity. |
| _graph_input | بررسی dtype index، transpose یال و ساخت ModelInput گراف. |
| _evaluate_graphs | cat داده‌ها، offset یال، یک اجرای عددی و split/reconstruct ساختار عمومی. |
| evaluate_contract | انتخاب مسیر graph، core-vmap یا whole-batch و بازگرداندن Tensor/Value مناسب. |

### normalization.py

| تابع/کلاس | مسئولیت افزوده |
|---|---|
| _mask | ContextVar ماسک فراخوانی؛ state دائمی batch روی شبکه نیست. |
| normalization_mask | ورود/خروج امن scope، حفظ context قبلی و پاک‌سازی در exception. |
| has_normalization_mask | تشخیص scope فعال برای تبدیل ماسک در سطح‌های PointNet++. |
| batch_norms | اسکن زیرماژول‌های واقعی از خانوادهٔ native _BatchNorm. |
| uses_batch_statistics | تشخیص training یا نبود هر دو running buffer؛ eval بدون tracked stats هنوز coupled است. |
| masked_point_inputs | صفرکردن ورودی نامعتبر پیش از convolution برای جلوگیری از آلودگی NaN در گرادیان. |
| _checked_mask | بررسی bool، شکل فضایی و device ماسک. |
| _BatchNorm.forward | dispatch Tensor/list، gather نمونهٔ معتبر، یک BN native و scatter/restore. |
| BatchNorm1d/2d/3d | subclassهای قابل استفادهٔ مستقل با پارامترها و state native. |
| InstanceNorm1d.forward | مسیر native بدون mask، پردازش مستقل list و محاسبهٔ per-case ماسک‌دار. |
| point_normalization | سیاست batch/none/instance PointNet و حذف norm پس از pooling در instance. |

فایل _point_batches.py در یک نسخهٔ میانی برای padding خودکار list افزوده شده بود؛ اکنون حذف شده است و قابلیت جاری محسوب نمی‌شود. اشاره‌های قدیمی به ragged PointNet با BN باید با قرارداد بند ۶.۳ جایگزین شوند.

## ۱۱. هزینه‌ها و تفاوت‌هایی که صرفاً از ظاهر API معلوم نیستند

| مورد | اثر واقعی | وضعیت/علت |
|---|---|---|
| validation پیش از vmap | آماده‌سازی/reshape/broadcast برای validation و سپس دوباره برای evaluate؛ list recursion می‌تواند validation بیشتری بسازد. | برای دور نگه‌داشتن شرط داده‌محور از vmap؛ حذف کامل overhead انجام نشده است. |
| انتخاب خروجی | head/auxiliary انتخاب‌نشده ممکن است باز هم محاسبه شود. | فقط تحویل خروجی محدود شده، نه pipeline داخلی. |
| اسکن signature | signature با توجه به ماژول‌های واقعی و train/eval ساخته می‌شود؛ اسکن BN یک کار مدیریتی Python است. | caching این metadata افزوده نشده است. |
| GINO mask ثابت | محاسبه روی slot نامعتبر و تغییر احتمالی chunk size. | امکان vmap در برابر boolean slicing با طول داده‌محور. |
| checkpoint بزرگ‌تر GINO | saved tensor کمتر، replay گسترده‌تر. | checkpoint بیرون vmap؛ تضمین speedup نداریم. |
| packing گراف | cat، offset و split/stack با هزینهٔ حافظه/مدیریت. | اجرای یک گراف disjoint بدون بازنویسی message passing. |
| norm لیستی/ماسک‌دار | gather/cat/scatter و dynamic indexing. | محاسبهٔ آمار واقعی بدون padding آماری. |
| DeepNURBS endpoint | ساخت endpoint_basis و where حتی بدون endpoint راست. | حذف branch داده‌محور برای vmap. |
| PointNet++ FPS | نگهداری list index و stack به جای بافر inplace. | سازگاری vmap؛ الگوریتم انتخاب نقاط حفظ شده است. |

هیچ ادعایی نداریم که torch.compile همهٔ این کارهای مدیریتی یا محاسبات انتخاب‌نشده را حتماً حذف می‌کند. در این کار graph scheduler جدید، thread pool، job موازی geometry، CUDA kernel اختصاصی یا backend مشتق سریع برای معماری‌های خاص اضافه نشده است.

## ۱۲. آزمون‌ها و حدود نتیجه‌گیری

۱. در همین بررسی، source اصلی بدون تغییر از snapshot 80e5c8d در یک پوشهٔ موقت با نام پکیج models استخراج شد. برای هر ۱۱ شبکه، یک config نمایندهٔ کوچک در eval، state_dict یکسان و ورودی عددی یکسان استفاده شد. predictions، latent موجود، بارگذاری strict کلیدهای وزن و گرادیان پارامترها مقایسه شدند؛ هر ۱۱ مورد گذشتند. این مقایسه، oracle مستقل upstream داشت و صرفاً مقایسهٔ دو متد نسخهٔ فعلی نبود. اسکریپت یک‌باره در .build-tmp/audit_network_parity.py اجرا شد و جزو API/تست نصب‌شدهٔ پکیج نیست.

۲. تست دائمی test_all_networks_are_native_models_and_match_numerical_algorithm در tests/test_networks.py، مسیر Value را با predict همان نسخهٔ فعلی مقایسه می‌کند. این تست صحت bridge/layout را نشان می‌دهد؛ به‌تنهایی شاهد مستقلِ برابری کامل با upstream نیست.

۳. AST بدنهٔ ۱۴ متد با snapshot اصلی برابر بررسی شد: PointNetEncoder.forward، TransformNet.forward، دو head PointNet، GCNConv/GCNBlock/GATConv/GATBlock.forward، MeshGraphNetBlock.forward، FourierOperatorBlock3d/ProjectionHead3d.forward و ConvBlock3D/EncoderBlock3D/DecoderBlock3D.forward. برابر بودن بدنه، به معنی برابر بودن constructor یا کلاس norm استفاده‌شده نیست؛ آن تغییرها جدا توضیح داده شدند.

۴. تست‌های دائمی normalization، مسیر native، list، mask، خروجی، running buffers، momentum=None و گرادیان/مشتق دوم را بررسی می‌کنند. تست‌های dense network نیز norm مشترک را با native جایگزین می‌کنند و مقدار خروجی، latent، state_dict و bufferها را مقایسه می‌کنند. آمار mask-aware و حالت instance عمداً با رفتار ماسک‌دار قبلی یکسان فرض نمی‌شوند.

۵. تست‌های شبکه شامل batching چندمحوری، حفظ list برای مدل‌های مجاز، packing گراف، اتصال واقعی GeoCore تا MeshGraphNet در Solver، قواعد استقلال، dropout و checkpoint مشتق‌های بالاتر GINO هستند. قرارداد جدیدِ رد list در PointNetهای دارای BN باید در تست‌ها نیز بررسی شود؛ تست adapter padding حذف‌شده شاهد قابلیت فعلی نیست.

۶. پس از تغییرهای همین نوبت، کل مجموعهٔ ۲۵۷ تست روی CPU گذشت؛ یک FutureWarning موجود مربوط به TorchScript روی Python 3.14 باقی است. تست‌های GeoCore به‌تنهایی ۱۹ مورد و تست‌های شبکه‌ها ۱۰۰ مورد بودند. مثال geocore_design.py نیز با مختصات dense به شکل (2,32,3) و centroid به شکل (2,3) اجرا شد. این نتایج جای ادعای پوشش تمام configهای upstream را نمی‌گیرند.

همهٔ این شواهد نمونه‌محورند. تمام configها، تمام maskها، GPU، کارایی fullgraph compilation و برابری بیت‌به‌بیت training با dropout پوشش داده نشده‌اند. برابر بودن خروجی نمونه به معنی حفظ همهٔ تصمیمات اجرایی نیست.

## ۱۳. تصمیم‌هایی که بهتر است مشخصاً با همکار مرور شوند

این موارد لزوماً اشتباه نیستند؛ نیازمند آگاهی و توافق‌اند چون به ساختار یا معنای اجرای قبلی او مربوط می‌شوند:

1. **وابستگی مستقیم شبکه‌ها به موتور:** آیا حفظ BaseNetwork مستقل و یک bridge جدا بهتر است، یا native Model بودن اولویت دارد؟ API فعلی عمداً constructor/forward قدیمی را حفظ نکرده است.
2. **قرارداد مرکزی مبتنی بر نام شبکه:** آیا دانستن _specs در یک فایل مرکزی پذیرفتنی است، یا قرارداد هر معماری باید کنار خودش باشد؟ در نسخهٔ فعلی extension فقط registryمحور نیست.
3. **تعریف batch در گراف:** آیا BN باید آمار تمام رأس‌ها را مشترک بگیرد، یا هر گراف وزن/آمار مستقل داشته باشد؟ packing فعلی حالت اول را انتخاب می‌کند؛ MGN با LayerNorm موضوع متفاوتی است.
4. **معنای mask در normalization:** نسخهٔ ما ردیف نامعتبر را از آمار خارج می‌کند. برای PointNet++ وزن تکرار همسایهٔ واقعی حفظ شده است؛ تغییر آن به حذف تمام neighbor slotهای نامعتبر تصمیم دیگری خواهد بود.
5. **حالت instance PointNet:** حذف norm بعد از pooling و فعال‌شدن bias آن FCها انتخاب تازهٔ این پروژه است؛ نباید صرفاً rename گزینهٔ قدیمی تلقی شود.
6. **محدودهٔ checkpoint GINO و masked work:** حفظ batching در برابر replay بزرگ‌تر و محاسبه روی slotهای نامعتبر، نیازمند benchmark متناسب با دادهٔ واقعی است.
7. **API عددی و hookها:** اگر ابزارهای همکار به forward hooks سطح شبکه یا torch.compile(network) در مسیر ModelInput تکیه دارند، predict فعلی جایگزین کاملاً شفاف نیست.
8. **helper عمومی FPS:** حذف اعتبارسنجی standalone برای vmap، بهتر است بعداً با یک مرز public/internal روشن‌تر جبران شود.
9. **کدهای مرجع خارج از models:** adaptation، rollout و مثال‌های اصلی هنوز با قرارداد موتور بازنویسی نشده‌اند؛ دربارهٔ فعال‌کردن آن‌ها تصمیم جدا لازم است.

## ۱۴. تغییر مرز اتصال GeoCore در همین نوبت

این مورد جزو تغییر source Neural3DBench نیست، ولی روی داده‌ای که شبکه دریافت می‌کند اثر دارد. GeoCoreModel قبلاً همهٔ خروجی‌ها را به list تبدیل می‌کرد؛ اکنون خروجی‌هایی که طبق source/options اندازهٔ ثابت دارند Tensor می‌شوند و دادهٔ واقعاً متغیر list می‌ماند. مثلاً point cloud با تعداد مشترک N، مستقیماً (S,N,3) می‌دهد؛ mesh با اندازهٔ متغیر list است حتی اگر دو عضو فعلاً برابر شوند. رپر همچنان هندسه‌ها را جدا روی CPU تولید می‌کند و بعد فقط خروجی ثابت را stack می‌کند.

بنابراین مسیر معمول point cloud ثابت می‌تواند Tensor را مستقیم به PointNet دارای BN بدهد. این به معنی padding یا تشخیص نوع از برابری اتفاقی شکل یک batch نیست. جزئیات voxel با spacing وابسته به bounds و NURBS پارامتری در docs/geocore.fa.md توضیح داده شده‌اند.

## ۱۵. فهرست کامل فایل‌های source و diff دقیق

در جدول بعدی، مسیرها نسبت به models منبع و networks فعلی‌اند. «یکسان» یعنی متن Python با نادیده‌گرفتن پایان خط تغییری نکرده است. تغییر import به‌تنهایی در ستون آخر مشخص شده؛ تغییر عمومی به فصل ۳ اشاره دارد. سپس diff دقیق تمام فایل‌های تغییرکرده و محتوای دو فایل افزوده آمده است تا جزئیات کوچک هم قابل بررسی باشند.

| فایل | وضعیت | توضیح |
|---|---|---|
| `__init__.py` | یکسان | بدون تغییر در متن Python. |
| `base.py` | تغییرکرده | کلاس پایه، binding اجباری، signature/evaluate، forward و predict جدید؛ فصل ۳. |
| `configs.py` | یکسان | بدون تغییر در متن Python. |
| `deep_nurbs/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `deep_nurbs/model.py` | تغییرکرده | قرارداد عمومی و where بدون branch برای endpoint؛ بند ۹.۵. |
| `fno/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `fno/components.py` | تغییرکرده | جایگزینی کلاس BatchNorm3d؛ forwardهای بلوک حفظ شده‌اند. |
| `fno/model.py` | تغییرکرده | قرارداد عمومی؛ pipeline عددی FNO حفظ شده است. |
| `fno/spectral.py` | یکسان | بدون تغییر در متن Python. |
| `gino/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `gino/components.py` | تغییرکرده | source_mask، طول ثابت، guard target خالی و انتخاب checkpoint؛ بندهای ۹.۳ و ۹.۴. |
| `gino/model.py` | تغییرکرده | قرارداد عمومی، source/query ماسک با طول ثابت، flag checkpoint و override predict. |
| `graph/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `graph/base.py` | تغییرکرده | فقط importهای نسبی؛ بدنهٔ کلاس بدون تغییر. |
| `graph/gat.py` | تغییرکرده | قرارداد عمومی؛ بدنهٔ attention اصلی حفظ شده است. |
| `graph/gcn.py` | تغییرکرده | قرارداد عمومی؛ محاسبهٔ GCN و عدم ارائهٔ latent حفظ شده‌اند. |
| `graph/layers.py` | تغییرکرده | جایگزینی کلاس BatchNorm1d در GCNBlock و GATBlock؛ forwardها حفظ شده‌اند. |
| `graph/meshgraphnet/__init__.py` | یکسان | بدون تغییر در متن Python. |
| `graph/meshgraphnet/blocks.py` | یکسان | بدون تغییر در متن Python. |
| `graph/meshgraphnet/decoder.py` | یکسان | بدون تغییر در متن Python. |
| `graph/meshgraphnet/encoder.py` | یکسان | بدون تغییر در متن Python. |
| `graph/meshgraphnet/model.py` | تغییرکرده | قرارداد عمومی؛ encoder/processor/decoder اصلی حفظ شده‌اند. |
| `graph/meshgraphnet/processor.py` | یکسان | بدون تغییر در متن Python. |
| `pointnet/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `pointnet/components.py` | تغییرکرده | factory norm، instance قبل از pooling، انتخاب norm/bias پس از pooling. |
| `pointnet/model.py` | تغییرکرده | قرارداد عمومی، گزینهٔ instance، پاک‌کردن ورودی نامعتبر و scope mask. |
| `pointnet2/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `pointnet2/components.py` | تغییرکرده | FPS list+stack، جابه‌جایی validation، BN مشترک و mask در grouping/propagation. |
| `pointnet2/model.py` | تغییرکرده | قرارداد عمومی، پاک‌کردن ورودی نامعتبر و scope mask. |
| `registry.py` | تغییرکرده | inputs/outputs اجباری در create و انتقال به constructor؛ منطق ثبت حفظ شده است. |
| `representations.py` | یکسان | بدون تغییر در متن Python. |
| `siren/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `siren/layers.py` | یکسان | بدون تغییر در متن Python. |
| `siren/model.py` | تغییرکرده | قرارداد عمومی؛ SineLayer و محاسبهٔ عددی حفظ شده‌اند. |
| `transolver/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `transolver/block.py` | یکسان | بدون تغییر در متن Python. |
| `transolver/components.py` | یکسان | بدون تغییر در متن Python. |
| `transolver/model.py` | تغییرکرده | قرارداد عمومی؛ pipeline attention، auxiliary و initialization حفظ شده‌اند. |
| `transolver/physics_attention.py` | یکسان | بدون تغییر در متن Python. |
| `types.py` | یکسان | بدون تغییر در متن Python. |
| `unet3d/__init__.py` | تغییرکرده | فقط import مربوط به registration از models.registry به مسیر نسبی تبدیل شده است. |
| `unet3d/components.py` | تغییرکرده | فقط جایگزینی BatchNorm3d در factory؛ بقیهٔ گزینه‌ها و forwardها حفظ شده‌اند. |
| `unet3d/model.py` | تغییرکرده | قرارداد عمومی؛ pipeline و encoder/decoder اصلی حفظ شده‌اند. |

دو فایل Python جدید فعال نسبت به upstream: `_contract.py` و `normalization.py`. فایل `_point_batches.py` فقط در نسخهٔ میانی موتور وجود داشت و در قرارداد فعلی حذف شده است.

### پیوست: diff کامل source

این diff با نرمال‌کردن پایان خط و فاصلهٔ انتهای خط برای نمایش Markdown، و بدون حذف هیچ hunk محتوایی از snapshot مشخص‌شده تولید شده است. شمارهٔ hunkها به همین snapshot تعلق دارند؛ با تغییرات بعدی ممکن است جابه‌جا شوند. فایل‌های یکسان جدول بالا diff ندارند. دو فایل افزوده نیز کامل آمده‌اند. تغییرهای خود Model موتور، Solver و GeoCore در این پیوست source شبکه‌ها نیستند؛ اثرهای مرتبطشان در متن توضیح داده شده است.

<details>
<summary>base.py</summary>

```diff
--- third_party/neural3dbench/models/base.py
+++ src/pinn_engine/networks/base.py
@@ -1,23 +1,58 @@
 from __future__ import annotations

 from abc import ABC, abstractmethod
-from typing import Any
+from typing import Any, Mapping

 import torch
-from torch import nn
+
+from pinn_engine.model import Model, Signature
+from pinn_engine.quantities import Quantity
+from pinn_engine.values import Value

 from .configs import BaseModelConfig
 from .types import ModelCapabilities, ModelInput, ModelOutput


-class BaseNetwork(nn.Module, ABC):
-    """Stable interface every model must satisfy."""
+class BaseNetwork(Model, ABC):
+    """Native named-quantity model with the original numeric prediction API."""

     config_class = BaseModelConfig
+    # Dropout draws independently for automatically mapped batch members.
+    vmap_randomness = "different"

-    def __init__(self, config: BaseModelConfig | None = None):
+    def __init__(
+        self,
+        config: BaseModelConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
         super().__init__()
         self.config = config or self.config_class()
+        from ._contract import initialize_contract
+
+        initialize_contract(self, inputs, outputs)
+
+    @property
+    def signature(self) -> Signature:
+        """Derive axes and independence from the actual network modules."""
+        from ._contract import contract_signature
+
+        return contract_signature(self)
+
+    def evaluate(
+        self, values: Mapping[Quantity, torch.Tensor | list[torch.Tensor] | Value]
+    ) -> dict[Quantity, torch.Tensor | list[torch.Tensor] | Value]:
+        from ._contract import evaluate_contract
+
+        return evaluate_contract(self, values)
+
+    def forward(self, values: Mapping[Quantity, Value]) -> dict[Quantity, Value]:
+        """Validate tensor data before entering automatic vectorization."""
+        from ._contract import validate_contract
+
+        validate_contract(self, values)
+        return super().forward(values)

     @property
     @abstractmethod
@@ -62,11 +97,13 @@
         return True

     @abstractmethod
-    def forward(self, inputs: ModelInput) -> ModelOutput:
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         raise NotImplementedError

     def predict(self, inputs: ModelInput) -> ModelOutput:
-        return self(inputs)
+        """Run the numeric API, retaining auxiliary and structured outputs."""
+        self.validate_inputs(inputs)
+        return self._predict(inputs)

     @staticmethod
     def _ensure_tensor(value: Any, *, name: str = "tensor") -> torch.Tensor:
```

</details>

<details>
<summary>deep_nurbs/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/deep_nurbs/__init__.py
+++ src/pinn_engine/networks/deep_nurbs/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .model import DeepNURBS, DeepNURBSConfig

```

</details>

<details>
<summary>deep_nurbs/model.py</summary>

```diff
--- third_party/neural3dbench/models/deep_nurbs/model.py
+++ src/pinn_engine/networks/deep_nurbs/model.py
@@ -4,12 +4,14 @@
 from numbers import Integral
 from typing import Any, Mapping, Sequence

+from pinn_engine.quantities import Quantity
+
 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput


 def _positive_int(value: int, name: str) -> int:
@@ -224,10 +226,10 @@

     basis = basis[..., :control_count]
     at_right_endpoint = values == knots[-degree - 1]
-    if torch.any(at_right_endpoint):
-        endpoint_basis = torch.zeros_like(basis)
-        endpoint_basis[..., -1] = 1.0
-        basis = torch.where(at_right_endpoint.unsqueeze(-1), endpoint_basis, basis)
+    # Tensor selection also supports mapped inputs at the right endpoint.
+    endpoint_basis = torch.zeros_like(basis)
+    endpoint_basis[..., -1] = 1.0
+    basis = torch.where(at_right_endpoint.unsqueeze(-1), endpoint_basis, basis)
     return basis


@@ -236,8 +238,14 @@

     config_class = DeepNURBSConfig

-    def __init__(self, config: DeepNURBSConfig | None = None) -> None:
-        super().__init__(config=config or DeepNURBSConfig())
+    def __init__(
+        self,
+        config: DeepNURBSConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or DeepNURBSConfig(), inputs=inputs, outputs=outputs)
         control_points = torch.tensor(self.config.control_points, dtype=torch.float32)
         weights = torch.tensor(self.config.weights, dtype=torch.float32)
         admissible = torch.tensor(
@@ -323,8 +331,7 @@
         )
         return physical_coordinates, admissible_field, lifting_field

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         parametric_coordinates = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
```

</details>

<details>
<summary>fno/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/fno/__init__.py
+++ src/pinn_engine/networks/fno/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .components import FourierOperatorBlock3d, ProjectionHead3d, normalized_coordinate_grid
 from .model import FNO3D, FNO3DConfig
```

</details>

<details>
<summary>fno/components.py</summary>

```diff
--- third_party/neural3dbench/models/fno/components.py
+++ src/pinn_engine/networks/fno/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from pinn_engine.networks.normalization import BatchNorm3d

 from .spectral import SpectralConv3d

@@ -49,7 +50,7 @@
         if normalization == "none":
             self.normalization: nn.Module = nn.Identity()
         elif normalization == "batch":
-            self.normalization = nn.BatchNorm3d(width)
+            self.normalization = BatchNorm3d(width)
         elif normalization == "instance":
             self.normalization = nn.InstanceNorm3d(width, affine=True)
         else:
```

</details>

<details>
<summary>fno/model.py</summary>

```diff
--- third_party/neural3dbench/models/fno/model.py
+++ src/pinn_engine/networks/fno/model.py
@@ -1,15 +1,18 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity
 from typing import Sequence

 import torch
 from torch import nn
 from torch.nn import functional as F

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .components import (
     FourierOperatorBlock3d,
@@ -77,8 +80,14 @@

     config_class = FNO3DConfig

-    def __init__(self, config: FNO3DConfig | None = None) -> None:
-        super().__init__(config=config or FNO3DConfig())
+    def __init__(
+        self,
+        config: FNO3DConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or FNO3DConfig(), inputs=inputs, outputs=outputs)
         lifted_channels = self.config.input_channels + (
             3 if self.config.append_coordinates else 0
         )
@@ -134,8 +143,7 @@
         self.blocks[0].spectral._validate_spatial_shape(volume.shape[-3:])
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         x = torch.as_tensor(
             inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
```

</details>

<details>
<summary>gino/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/gino/__init__.py
+++ src/pinn_engine/networks/gino/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .components import KernelIntegralTransform
 from .model import GINO, GINOConfig
```

</details>

<details>
<summary>gino/components.py</summary>

```diff
--- third_party/neural3dbench/models/gino/components.py
+++ src/pinn_engine/networks/gino/components.py
@@ -35,8 +35,31 @@
         source_coordinates: torch.Tensor,
         target_coordinates: torch.Tensor,
         source_features: torch.Tensor,
+        source_mask: torch.Tensor | None = None,
+        *,
+        use_checkpoint: bool = True,
     ) -> torch.Tensor:
-        if source_coordinates.shape[0] == 0:
+        """Integrate valid sources without data-dependent Tensor indexing.
+
+        A mask keeps the source shape fixed so independently batched calls can
+        be transformed with ``vmap``. Invalid source coordinates and features
+        may contain padding values, including NaNs, and do not contribute.
+        Callers using functional transforms disable checkpointing explicitly;
+        PyTorch checkpoint's saved-tensor hooks do not support those transforms.
+        """
+        if source_mask is not None:
+            if source_mask.shape != source_coordinates.shape[:1]:
+                raise ValueError("source_mask must have shape [N].")
+            if source_mask.dtype != torch.bool:
+                raise TypeError("source_mask must use a boolean dtype.")
+            source_coordinates = torch.where(
+                source_mask.unsqueeze(-1), source_coordinates, 0.0
+            )
+            source_features = torch.where(
+                source_mask.unsqueeze(-1), source_features, 0.0
+            )
+
+        if source_coordinates.shape[0] == 0 or target_coordinates.shape[0] == 0:
             return source_features.new_zeros(
                 (target_coordinates.shape[0], self.source_projection.out_features)
             )
@@ -55,15 +78,19 @@
             query = target_coordinates[
                 query_start : query_start + query_chunk_size
             ]
-            integrate = lambda query_chunk, sources, values: self._integrate_query_chunk(
-                query_chunk, sources, values, source_chunk_size
-            )
-            if torch.is_grad_enabled():
+
+            def integrate(query_chunk, sources, values, mask):
+                return self._integrate_query_chunk(
+                    query_chunk, sources, values, source_chunk_size, mask
+                )
+
+            if use_checkpoint and torch.is_grad_enabled():
                 output = checkpoint(
-                    integrate, query, source_coordinates, projected, use_reentrant=False
+                    integrate, query, source_coordinates, projected, source_mask,
+                    use_reentrant=False,
                 )
             else:
-                output = integrate(query, source_coordinates, projected)
+                output = integrate(query, source_coordinates, projected, source_mask)
             outputs.append(output)

         return torch.cat(outputs, dim=0)
@@ -74,6 +101,7 @@
         source_coordinates: torch.Tensor,
         projected: torch.Tensor,
         source_chunk_size: int,
+        source_mask: torch.Tensor | None = None,
     ) -> torch.Tensor:
         numerator = projected.new_zeros((query.shape[0], projected.shape[-1]))
         denominator = projected.new_zeros((query.shape[0], 1))
@@ -88,6 +116,8 @@
                 self.kernel(delta.reshape(-1, 3)).reshape_as(distance)
             )
             weights = cutoff * learned_kernel
+            if source_mask is not None:
+                weights = weights * source_mask[source_start:source_end].unsqueeze(0)
             numerator = numerator + weights @ projected[source_start:source_end]
             denominator = denominator + weights.sum(dim=1, keepdim=True)
         return numerator / denominator.clamp_min(1e-8)
```

</details>

<details>
<summary>gino/model.py</summary>

```diff
--- third_party/neural3dbench/models/gino/model.py
+++ src/pinn_engine/networks/gino/model.py
@@ -1,16 +1,19 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity
 from numbers import Integral
 from typing import Sequence

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.fno.components import FourierOperatorBlock3d, get_activation
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..fno.components import FourierOperatorBlock3d, get_activation
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .components import KernelIntegralTransform

@@ -93,9 +96,18 @@
     """Geometry-aware point-to-grid, Fourier, and grid-to-point operator."""

     config_class = GINOConfig
-
-    def __init__(self, config: GINOConfig | None = None) -> None:
-        super().__init__(config=config or GINOConfig())
+    # Model.forward checkpoints the complete call after building its vmaps.
+    # Inner kernel checkpoints stay off in evaluate to avoid nested transforms.
+    checkpoint_batches = True
+
+    def __init__(
+        self,
+        config: GINOConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or GINOConfig(), inputs=inputs, outputs=outputs)
         self.encoder = KernelIntegralTransform(
             self.config.input_dim,
             self.config.width,
@@ -204,8 +216,7 @@
         grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
         return grid.reshape(-1, 3)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput, *, use_checkpoint: bool = False) -> ModelOutput:
         parameter = next(self.parameters())
         coordinates = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
@@ -228,9 +239,11 @@
         for batch_index in range(coordinates.shape[0]):
             valid = source_mask[batch_index]
             latent = self.encoder(
-                coordinates[batch_index, valid],
+                coordinates[batch_index],
                 latent_coordinates,
-                features[batch_index, valid],
+                features[batch_index],
+                source_mask=valid,
+                use_checkpoint=use_checkpoint,
             )
             latent = latent.reshape(*self.config.latent_resolution, self.config.width)
             latent_batches.append(latent.permute(3, 0, 1, 2))
@@ -247,19 +260,18 @@
         decoded_batches = []
         for batch_index in range(coordinates.shape[0]):
             query_mask = source_mask[batch_index] if default_queries else None
-            valid_queries = queries[batch_index] if query_mask is None else queries[batch_index, query_mask]
+            valid_queries = queries[batch_index]
+            if query_mask is not None:
+                valid_queries = torch.where(query_mask.unsqueeze(-1), valid_queries, 0.0)
             decoded = self.decoder(
                 latent_coordinates,
                 valid_queries,
                 flat_latent[batch_index],
+                use_checkpoint=use_checkpoint,
             )
             projected = self.output_projection(decoded)
             if query_mask is not None:
-                batch_predictions = projected.new_zeros(
-                    (queries.shape[1], self.config.output_dim)
-                )
-                batch_predictions[query_mask] = projected
-                decoded_batches.append(batch_predictions)
+                decoded_batches.append(projected.masked_fill(~query_mask.unsqueeze(-1), 0.0))
             else:
                 decoded_batches.append(projected)

@@ -274,3 +286,8 @@
                 "latent_resolution": self.config.latent_resolution,
             },
         )
+
+    def predict(self, inputs: ModelInput) -> ModelOutput:
+        """Retain bounded-memory checkpointing outside functional transforms."""
+        self.validate_inputs(inputs)
+        return self._predict(inputs, use_checkpoint=True)
```

</details>

<details>
<summary>graph/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/graph/__init__.py
+++ src/pinn_engine/networks/graph/__init__.py
@@ -1,6 +1,6 @@
 from __future__ import annotations

-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .base import BaseGraphNetwork
 from .gat import GAT, GATConfig
```

</details>

<details>
<summary>graph/base.py</summary>

```diff
--- third_party/neural3dbench/models/graph/base.py
+++ src/pinn_engine/networks/graph/base.py
@@ -4,8 +4,8 @@

 import torch

-from models.base import BaseNetwork
-from models.types import ModelCapabilities
+from ..base import BaseNetwork
+from ..types import ModelCapabilities


 class BaseGraphNetwork(BaseNetwork, ABC):
```

</details>

<details>
<summary>graph/gat.py</summary>

```diff
--- third_party/neural3dbench/models/graph/gat.py
+++ src/pinn_engine/networks/graph/gat.py
@@ -1,12 +1,15 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch
 from torch import nn

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ..configs import BaseModelConfig
+from ..types import ModelInput, ModelOutput

 from .base import BaseGraphNetwork
 from .layers import GATBlock
@@ -84,8 +87,14 @@

     config_class = GATConfig

-    def __init__(self, config: GATConfig | None = None):
-        super().__init__(config=config or GATConfig())
+    def __init__(
+        self,
+        config: GATConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__(config=config or GATConfig(), inputs=inputs, outputs=outputs)
         self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
         self.layers = nn.ModuleList(
             GATBlock(
@@ -139,8 +148,7 @@
         self_loops = torch.stack((nodes, nodes), dim=0)
         return torch.cat((non_loop_edges, self_loops), dim=1)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(
             device=self.input_proj.weight.device,
             dtype=self.input_proj.weight.dtype,
```

</details>

<details>
<summary>graph/gcn.py</summary>

```diff
--- third_party/neural3dbench/models/graph/gcn.py
+++ src/pinn_engine/networks/graph/gcn.py
@@ -1,12 +1,15 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch
 from torch import nn

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ..configs import BaseModelConfig
+from ..types import ModelInput, ModelOutput

 from .base import BaseGraphNetwork
 from .layers import GCNBlock
@@ -53,8 +56,14 @@
 class GCN(BaseGraphNetwork):
     config_class = GCNConfig

-    def __init__(self, config: GCNConfig | None = None):
-        super().__init__(config=config or GCNConfig())
+    def __init__(
+        self,
+        config: GCNConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__(config=config or GCNConfig(), inputs=inputs, outputs=outputs)

         self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

@@ -105,8 +114,7 @@

         return edge_index

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:

         node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(self.input_proj.weight.device)
         edge_index = self._prepare_edge_index(inputs.edge_index, node_features.size(0))
```

</details>

<details>
<summary>graph/layers.py</summary>

```diff
--- third_party/neural3dbench/models/graph/layers.py
+++ src/pinn_engine/networks/graph/layers.py
@@ -2,6 +2,7 @@

 import torch
 from torch import nn
+from pinn_engine.networks.normalization import BatchNorm1d


 def _get_activation(name: str | None) -> nn.Module:
@@ -71,7 +72,7 @@
         if self.normalization == "layer_norm":
             self.norm = nn.LayerNorm(out_features)
         elif self.normalization == "batch_norm":
-            self.norm = nn.BatchNorm1d(out_features)
+            self.norm = BatchNorm1d(out_features)
         elif self.normalization == "instance_norm":
             self.norm = nn.InstanceNorm1d(out_features)
         else:
@@ -236,7 +237,7 @@
         if self.normalization == "layer_norm":
             self.norm = nn.LayerNorm(out_features)
         elif self.normalization == "batch_norm":
-            self.norm = nn.BatchNorm1d(out_features)
+            self.norm = BatchNorm1d(out_features)
         else:
             self.norm = None

```

</details>

<details>
<summary>graph/meshgraphnet/model.py</summary>

```diff
--- third_party/neural3dbench/models/graph/meshgraphnet/model.py
+++ src/pinn_engine/networks/graph/meshgraphnet/model.py
@@ -1,11 +1,14 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ...configs import BaseModelConfig
+from ...types import ModelInput, ModelOutput

 from ..base import BaseGraphNetwork
 from .decoder import MeshGraphNetDecoder
@@ -76,8 +79,14 @@

     config_class = MeshGraphNetConfig

-    def __init__(self, config: MeshGraphNetConfig | None = None) -> None:
-        super().__init__(config=config or MeshGraphNetConfig())
+    def __init__(
+        self,
+        config: MeshGraphNetConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or MeshGraphNetConfig(), inputs=inputs, outputs=outputs)
         common = {
             "mlp_hidden_dim": self.config.mlp_hidden_dim,
             "mlp_layers": self.config.mlp_layers,
@@ -140,8 +149,7 @@
             raise TypeError("edge_index must use an integer dtype.")
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         node_features = self._ensure_tensor(
             inputs.node_features, name="node_features"
```

</details>

<details>
<summary>pointnet/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet/__init__.py
+++ src/pinn_engine/networks/pointnet/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .components import (
     PointNetClassificationHead,
```

</details>

<details>
<summary>pointnet/components.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet/components.py
+++ src/pinn_engine/networks/pointnet/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from pinn_engine.networks.normalization import point_normalization


 def feature_transform_regularizer(transform: torch.Tensor) -> torch.Tensor:
@@ -53,8 +54,8 @@
                 )
             )
             if final_activation or not is_last:
-                if normalization == "batch":
-                    layers.append(nn.BatchNorm1d(out_channels))
+                if normalization != "none":
+                    layers.append(point_normalization(normalization, out_channels))
                 layers.append(nn.ReLU(inplace=True))
         self.layers = nn.Sequential(*layers)

@@ -71,10 +72,10 @@
         self.point_mlp = SharedPointMLP(
             (feature_dim, 64, 128, 1024), normalization=normalization
         )
-        self.fc1 = nn.Linear(1024, 512, bias=normalization == "none")
-        self.bn1 = nn.BatchNorm1d(512) if normalization == "batch" else nn.Identity()
-        self.fc2 = nn.Linear(512, 256, bias=normalization == "none")
-        self.bn2 = nn.BatchNorm1d(256) if normalization == "batch" else nn.Identity()
+        self.fc1 = nn.Linear(1024, 512, bias=normalization != "batch")
+        self.bn1 = point_normalization(normalization, 512, pooled=True)
+        self.fc2 = nn.Linear(512, 256, bias=normalization != "batch")
+        self.bn2 = point_normalization(normalization, 256, pooled=True)
         self.fc3 = nn.Linear(256, feature_dim * feature_dim)
         self.activation = nn.ReLU(inplace=True)

@@ -162,19 +163,19 @@
     ):
         super().__init__()
         layers: list[nn.Module] = [
-            nn.Linear(global_dim, 512, bias=normalization == "none")
+            nn.Linear(global_dim, 512, bias=normalization != "batch")
         ]
         if normalization == "batch":
-            layers.append(nn.BatchNorm1d(512))
+            layers.append(point_normalization(normalization, 512, pooled=True))
         layers.extend(
             (
                 nn.ReLU(inplace=True),
                 nn.Dropout(dropout),
-                nn.Linear(512, 256, bias=normalization == "none"),
+                nn.Linear(512, 256, bias=normalization != "batch"),
             )
         )
         if normalization == "batch":
-            layers.append(nn.BatchNorm1d(256))
+            layers.append(point_normalization(normalization, 256, pooled=True))
         layers.extend(
             (nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, output_dim))
         )
```

</details>

<details>
<summary>pointnet/model.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet/model.py
+++ src/pinn_engine/networks/pointnet/model.py
@@ -1,12 +1,16 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity
+from pinn_engine.networks.normalization import normalization_mask, masked_point_inputs

 import torch

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .components import (
     PointNetClassificationHead,
@@ -43,8 +47,8 @@
             raise ValueError("global_dim must be positive.")
         if self.task not in {"classification", "segmentation"}:
             raise ValueError("task must be 'classification' or 'segmentation'.")
-        if self.normalization not in {"batch", "none"}:
-            raise ValueError("normalization must be batch or none.")
+        if self.normalization not in {"batch", "instance", "none"}:
+            raise ValueError("normalization must be batch, instance, or none.")
         if not 0.0 <= self.dropout < 1.0:
             raise ValueError("dropout must be in the range [0, 1).")

@@ -58,8 +62,14 @@

     config_class = PointNetConfig

-    def __init__(self, config: PointNetConfig | None = None):
-        super().__init__(config=config or PointNetConfig())
+    def __init__(
+        self,
+        config: PointNetConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__(config=config or PointNetConfig(), inputs=inputs, outputs=outputs)
         self.encoder = PointNetEncoder(
             self.config.input_dim,
             self.config.global_dim,
@@ -139,8 +149,7 @@
                 raise ValueError("Every point cloud must contain at least one valid point.")
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         points = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
@@ -154,13 +163,15 @@
         if inputs.point_mask is not None:
             mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)

-        local, global_feature, transforms = self.encoder(points, features, mask)
-        if self.config.task == "classification":
-            predictions = self.head(global_feature)
-        else:
-            predictions = self.head(local, global_feature)
-            if mask is not None:
-                predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
+        points, features = masked_point_inputs(points, features, mask)
+        with normalization_mask(mask, enabled=mask is not None):
+            local, global_feature, transforms = self.encoder(points, features, mask)
+            if self.config.task == "classification":
+                predictions = self.head(global_feature)
+            else:
+                predictions = self.head(local, global_feature)
+                if mask is not None:
+                    predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)

         return ModelOutput(
             predictions=predictions,
```

</details>

<details>
<summary>pointnet2/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet2/__init__.py
+++ src/pinn_engine/networks/pointnet2/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .components import (
     FeaturePropagation,
```

</details>

<details>
<summary>pointnet2/components.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet2/components.py
+++ src/pinn_engine/networks/pointnet2/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from pinn_engine.networks.normalization import BatchNorm2d, normalization_mask, has_normalization_mask


 def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
@@ -43,22 +44,21 @@
             (batch_size, point_count), dtype=torch.bool, device=points.device
         )
     valid_counts = mask.sum(dim=1)
-    if torch.any(valid_counts == 0):
-        raise ValueError("Every point cloud must contain at least one valid point.")
+    # The public model entry validates nonempty masks before vmap starts.

     centroid = (points * mask.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
     initial_distance = (points - centroid.unsqueeze(1)).square().sum(dim=-1)
     farthest = initial_distance.masked_fill(~mask, -1.0).argmax(dim=1)

-    indices = torch.empty(
-        (batch_size, sample_count), dtype=torch.long, device=points.device
-    )
+    indices = []
     minimum_distance = torch.full(
         (batch_size, point_count), float("inf"), device=points.device
     )
     batch_indices = torch.arange(batch_size, device=points.device)
     for sample_index in range(sample_count):
-        indices[:, sample_index] = farthest
+        # Stack tensor results rather than copying mapped data into an
+        # unbatched allocation, which is not supported by torch.func.vmap.
+        indices.append(farthest)
         selected = points[batch_indices, farthest]
         distance = (points - selected.unsqueeze(1)).square().sum(dim=-1)
         minimum_distance = torch.minimum(minimum_distance, distance)
@@ -68,7 +68,7 @@
         torch.arange(sample_count, device=points.device).unsqueeze(0)
         < valid_counts.unsqueeze(1).clamp_max(sample_count)
     )
-    return indices, sampled_mask
+    return torch.stack(indices, dim=1), sampled_mask


 def query_ball_point(
@@ -145,7 +145,7 @@
                 )
             )
             if normalization == "batch":
-                layers.append(nn.BatchNorm2d(output_channels))
+                layers.append(BatchNorm2d(output_channels))
             layers.append(nn.ReLU(inplace=True))
         self.layers = nn.Sequential(*layers)

@@ -218,7 +218,12 @@
                 grouped_features = index_points(features, group_indices)
             grouped = torch.cat((relative_points, grouped_features), dim=-1)

-        encoded = self.mlp(grouped.permute(0, 3, 2, 1))
+        # Original radius grouping repeats a real neighbor in unused slots;
+        # retain that weighting. Only added padding/invalid centroids are new.
+        norm_mask = (grouped_mask.permute(0, 2, 1) if self.group_all else
+                     centroid_mask[:, None, :].expand(-1, grouped.shape[2], -1))
+        with normalization_mask(norm_mask, enabled=has_normalization_mask()):
+            encoded = self.mlp(grouped.permute(0, 3, 2, 1))
         encoded = encoded.masked_fill(
             ~grouped_mask.permute(0, 2, 1).unsqueeze(1),
             torch.finfo(encoded.dtype).min,
@@ -239,7 +244,7 @@
         normalization: str = "batch",
     ):
         super().__init__()
-        from models.pointnet.components import SharedPointMLP
+        from ..pointnet.components import SharedPointMLP

         self.mlp = SharedPointMLP(
             (input_channels, *mlp_channels), normalization=normalization
@@ -269,7 +274,8 @@
         combined = interpolated
         if target_features is not None:
             combined = torch.cat((target_features, interpolated), dim=-1)
-        propagated = self.mlp(combined.transpose(1, 2)).transpose(1, 2)
+        with normalization_mask(target_mask, enabled=has_normalization_mask()):
+            propagated = self.mlp(combined.transpose(1, 2)).transpose(1, 2)
         return propagated.masked_fill(~target_mask.unsqueeze(-1), 0.0)


```

</details>

<details>
<summary>pointnet2/model.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet2/model.py
+++ src/pinn_engine/networks/pointnet2/model.py
@@ -1,13 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity
+from pinn_engine.networks.normalization import normalization_mask, masked_point_inputs

 import torch

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.pointnet.components import PointNetClassificationHead
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..pointnet.components import PointNetClassificationHead
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .components import PointNet2Encoder, PointNet2SegmentationHead

@@ -81,8 +85,14 @@

     config_class = PointNet2Config

-    def __init__(self, config: PointNet2Config | None = None):
-        super().__init__(config=config or PointNet2Config())
+    def __init__(
+        self,
+        config: PointNet2Config | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__(config=config or PointNet2Config(), inputs=inputs, outputs=outputs)
         self.encoder = PointNet2Encoder(
             self.config.feature_dim,
             self.config.sample_counts,
@@ -165,8 +175,7 @@
                 raise ValueError("Every point cloud must contain at least one valid point.")
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         points = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
@@ -182,20 +191,22 @@
             else torch.as_tensor(inputs.point_mask, device=points.device, dtype=torch.bool)
         )

-        global_feature, level_points, level_features, level_masks = self.encoder(
-            points, features, mask
-        )
-        if self.config.task == "classification":
-            predictions = self.head(global_feature)
-        else:
-            predictions = self.head(
-                points,
-                features,
-                mask,
-                level_points,
-                level_features,
-                level_masks,
-            )
+        points, features = masked_point_inputs(points, features, mask if inputs.point_mask is not None else None)
+        with normalization_mask(mask, enabled=inputs.point_mask is not None):
+            global_feature, level_points, level_features, level_masks = self.encoder(
+                points, features, mask
+            )
+            if self.config.task == "classification":
+                predictions = self.head(global_feature)
+            else:
+                predictions = self.head(
+                    points,
+                    features,
+                    mask,
+                    level_points,
+                    level_features,
+                    level_masks,
+                )
         return ModelOutput(
             predictions=predictions,
             latent=global_feature,
```

</details>

<details>
<summary>registry.py</summary>

```diff
--- third_party/neural3dbench/models/registry.py
+++ src/pinn_engine/networks/registry.py
@@ -1,7 +1,9 @@
 from __future__ import annotations

 import inspect
-from typing import Any, get_args, get_origin, Union
+from typing import Any, Mapping, get_args, get_origin, Union
+
+from pinn_engine.quantities import Quantity

 from .base import BaseNetwork
 from .configs import BaseModelConfig
@@ -36,7 +38,15 @@
         return cls._config_types.get(name, BaseModelConfig)

     @classmethod
-    def create(cls, name: str, config: dict[str, Any] | BaseModelConfig | None = None, **kwargs: Any) -> BaseNetwork:
+    def create(
+        cls,
+        name: str,
+        config: dict[str, Any] | BaseModelConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+        **kwargs: Any,
+    ) -> BaseNetwork:
         model_class = cls.get(name)
         config_type = cls.get_config_class(name)

@@ -49,7 +59,7 @@
         elif config is not None and not isinstance(config, BaseModelConfig):
             raise TypeError("Model config must be a BaseModelConfig or dict.")

-        return model_class(config=config)
+        return model_class(config=config, inputs=inputs, outputs=outputs)

     @classmethod
     def _infer_config_class(cls, model_class: type[BaseNetwork]) -> type[BaseModelConfig]:
```

</details>

<details>
<summary>siren/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/siren/__init__.py
+++ src/pinn_engine/networks/siren/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .layers import SineLayer
 from .model import SIREN, SIRENConfig
```

</details>

<details>
<summary>siren/model.py</summary>

```diff
--- third_party/neural3dbench/models/siren/model.py
+++ src/pinn_engine/networks/siren/model.py
@@ -2,13 +2,16 @@

 import math
 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .layers import SineLayer

@@ -49,8 +52,14 @@

     config_class = SIRENConfig

-    def __init__(self, config: SIRENConfig | None = None) -> None:
-        super().__init__(config=config or SIRENConfig())
+    def __init__(
+        self,
+        config: SIRENConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or SIRENConfig(), inputs=inputs, outputs=outputs)
         total_input_dim = self.config.spatial_dim + self.config.input_dim
         self.first_layer = SineLayer(
             total_input_dim,
@@ -132,8 +141,7 @@
                 raise ValueError("Every sample must contain at least one valid point.")
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         coordinates = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
```

</details>

<details>
<summary>transolver/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/transolver/__init__.py
+++ src/pinn_engine/networks/transolver/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .block import TransolverBlock
 from .components import FeedForward, InputEmbedding, OutputHead
```

</details>

<details>
<summary>transolver/model.py</summary>

```diff
--- third_party/neural3dbench/models/transolver/model.py
+++ src/pinn_engine/networks/transolver/model.py
@@ -1,13 +1,16 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .block import TransolverBlock
 from .components import InputEmbedding, OutputHead
@@ -54,8 +57,14 @@

     config_class = TransolverConfig

-    def __init__(self, config: TransolverConfig | None = None) -> None:
-        super().__init__(config=config or TransolverConfig())
+    def __init__(
+        self,
+        config: TransolverConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__(config=config or TransolverConfig(), inputs=inputs, outputs=outputs)
         self.input_embedding = InputEmbedding(
             self.config.spatial_dim,
             self.config.input_dim,
@@ -136,8 +145,7 @@
                 raise ValueError("Every sample must contain at least one valid point.")
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         coordinates = torch.as_tensor(
             inputs.coordinates, device=parameter.device, dtype=parameter.dtype
```

</details>

<details>
<summary>unet3d/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/unet3d/__init__.py
+++ src/pinn_engine/networks/unet3d/__init__.py
@@ -1,4 +1,4 @@
-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

 from .components import (
     ConvBlock3D,
```

</details>

<details>
<summary>unet3d/components.py</summary>

```diff
--- third_party/neural3dbench/models/unet3d/components.py
+++ src/pinn_engine/networks/unet3d/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from pinn_engine.networks.normalization import BatchNorm3d
 from torch.nn import functional as F


@@ -25,7 +26,7 @@

 def _normalization(name: str, channels: int) -> nn.Module:
     if name == "batch":
-        return nn.BatchNorm3d(channels)
+        return BatchNorm3d(channels)
     if name == "instance":
         return nn.InstanceNorm3d(channels, affine=True)
     if name == "group":
```

</details>

<details>
<summary>unet3d/model.py</summary>

```diff
--- third_party/neural3dbench/models/unet3d/model.py
+++ src/pinn_engine/networks/unet3d/model.py
@@ -1,13 +1,16 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from pinn_engine.quantities import Quantity

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from ..base import BaseNetwork
+from ..configs import BaseModelConfig
+from ..types import ModelCapabilities, ModelInput, ModelOutput

 from .components import DecoderBlock3D, EncoderBlock3D, VolumetricOutputHead

@@ -60,8 +63,14 @@

     config_class = UNet3DConfig

-    def __init__(self, config: UNet3DConfig | None = None):
-        super().__init__(config=config or UNet3DConfig())
+    def __init__(
+        self,
+        config: UNet3DConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__(config=config or UNet3DConfig(), inputs=inputs, outputs=outputs)
         channels = [
             self.config.base_channels * self.config.channel_multiplier**level
             for level in range(self.config.levels)
@@ -135,8 +144,7 @@
             )
         return True

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def _predict(self, inputs: ModelInput) -> ModelOutput:
         parameter = next(self.parameters())
         hidden = torch.as_tensor(
             inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
```

</details>

<details>
<summary>_contract.py — فایل افزوده</summary>

```diff
--- /dev/null
+++ src/pinn_engine/networks/_contract.py
@@ -0,0 +1,292 @@
+"""Named quantities and batch layouts for the native Neural3DBench models.
+
+The numerical networks retain their modules and state_dict names. This module
+only translates their channel-first/component conventions to engine Values.
+"""
+
+from __future__ import annotations
+
+from math import prod
+from typing import Mapping
+
+import torch
+
+from ..model import Signature
+from ..quantities import Quantity
+from ..values import Value, _check_value
+from .normalization import batch_norms, uses_batch_statistics
+
+
+_GRAPHS = {"gcn", "gat", "meshgraphnet"}
+_GRIDS = {"fno3d", "unet3d"}
+_SPATIAL = ("z", "y", "x")
+
+
+def _specs(model):
+    """Return role -> (core sample axes, component count), independently of B."""
+    c = model.config
+    name = c.name
+    if name in _GRAPHS:
+        width = c.node_input_dim if name == "meshgraphnet" else c.input_dim
+        inputs = {"node_features": (("vertex",), width), "edge_index": (("edge",), 2)}
+        if name == "meshgraphnet":
+            inputs["edge_features"] = (("edge",), c.edge_input_dim)
+        outputs = {"predictions": (("vertex",), c.output_dim)}
+        if name != "gcn":
+            outputs["latent"] = (("vertex",), c.latent_dim if name == "meshgraphnet" else c.hidden_dim)
+        return inputs, outputs, set(inputs)
+    if name in _GRIDS:
+        inputs = {"voxel_fields": (_SPATIAL, c.input_channels)}
+        latent_axes = _SPATIAL if name == "fno3d" else tuple("bottleneck_" + a for a in _SPATIAL)
+        width = c.width if name == "fno3d" else c.base_channels * c.channel_multiplier ** (c.levels - 1)
+        return inputs, {"predictions": (_SPATIAL, c.output_channels), "latent": (latent_axes, width)}, set(inputs)
+
+    spatial = 2 if name == "deep_nurbs" else getattr(c, "spatial_dim", 3)
+    inputs = {"coordinates": (("point",), spatial)}
+    required = {"coordinates"}
+    if name != "deep_nurbs":
+        features = c.feature_dim if name in {"pointnet", "pointnet2"} else c.input_dim
+        if features:
+            inputs["point_features"] = (("point",), features)
+            required.add("point_features")
+        inputs["point_mask"] = (("point",), 1)
+    point_axes = ("point",)
+    if name in {"pointnet", "pointnet2"}:
+        prediction_axes = () if c.task == "classification" else point_axes
+        outputs = {"predictions": (prediction_axes, c.output_dim), "latent": ((), c.global_dim)}
+    elif name == "gino":
+        inputs["query_coordinates"] = (("query",), 3)
+        prediction_axes = ("query",) if "query_coordinates" in model.input_quantities else point_axes
+        outputs = {"predictions": (prediction_axes, c.output_dim),
+                   "latent": (("latent_z", "latent_y", "latent_x"), c.width)}
+    else:
+        width = c.output_dim if name == "deep_nurbs" else c.hidden_dim
+        outputs = {"predictions": (point_axes, c.output_dim), "latent": (point_axes, width)}
+    return inputs, outputs, required
+
+
+def initialize_contract(model, inputs, outputs):
+    """Fix optional roles and check intrinsic component counts at construction."""
+    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
+        raise TypeError("inputs and outputs must map role names to Quantities")
+    model.input_quantities = dict(inputs)
+    model.output_quantities = dict(outputs)
+    input_specs, output_specs, required = _specs(model)
+    missing = required - set(inputs)
+    if missing:
+        raise ValueError(f"Missing required inputs: {sorted(missing)}")
+    if not outputs:
+        raise ValueError("Select at least one numeric output")
+    for side, bindings, specs in (("inputs", inputs, input_specs), ("outputs", outputs, output_specs)):
+        unknown = set(bindings) - set(specs)
+        if unknown:
+            raise ValueError(f"Unsupported {side}: {sorted(unknown)}")
+        if any(not isinstance(q, Quantity) for q in bindings.values()):
+            raise TypeError(f"{side} must contain Quantity objects")
+        if len(set(bindings.values())) != len(bindings):
+            raise ValueError(f"Each {side} role needs a distinct Quantity")
+        for role, q in bindings.items():
+            if prod(q.shape) != specs[role][1]:
+                raise ValueError(f"{side} {role!r}: Quantity shape {q.shape} needs {specs[role][1]} components")
+    if set(inputs.values()) & set(outputs.values()):
+        raise ValueError("Input and output Quantities must be distinct")
+    # Also reject different intrinsic shapes under one semantic Quantity name.
+    Signature({q: input_specs[r][0] for r, q in inputs.items()},
+              {q: output_specs[r][0] for r, q in outputs.items()})
+
+
+def contract_signature(model):
+    inputs, outputs, _ = _specs(model)
+    whole_batch = model.config.name in _GRAPHS or bool(batch_norms(model))
+    prefix = ("...",) if whole_batch else ()
+    incoming = {q: prefix + inputs[r][0] for r, q in model.input_quantities.items()}
+    outgoing = {q: prefix + outputs[r][0] for r, q in model.output_quantities.items()}
+    shared = {a for axes in incoming.values() for a in axes} & {a for axes in outgoing.values() for a in axes}
+    independent = model.config.name in {"siren", "deep_nurbs"}
+    rules = {a: independent for a in shared if a != "..."}
+    if "query" in shared:
+        # GINO queries do not feed its source encoder or one another.
+        rules["query"] = True
+    if "..." in shared:
+        # Frozen running statistics do not couple distinct batch members.
+        rules["..."] = not uses_batch_statistics(model)
+    return Signature(incoming, outgoing, rules)
+
+
+def _prepared_batches(model, values):
+    """Broadcast by axis name, flatten dense batches, retain outer ragged groups.
+
+    Each returned group has one upstream B dimension. No padding is introduced.
+    Missing named batch axes are shared, never matched by numerical size alone.
+    """
+    if model.config.name in {"pointnet", "pointnet2"} and batch_norms(model):
+        if any(isinstance(value.tensor, list) for value in values.values()):
+            raise ValueError(
+                f"{type(model).__name__} with BatchNorm requires Tensor inputs; "
+                "list inputs are not supported and are not padded automatically"
+            )
+    specs, output_specs, _ = _specs(model)
+    core = {a for roles, definitions in ((model.input_quantities, specs), (model.output_quantities, output_specs))
+            for role in roles for a in definitions[role][0]}
+    batch = tuple(dict.fromkeys(a for q in model.input_quantities.values()
+                               for a in values[q].axes if a not in core))
+    list_axes = {v.axes[0] for v in values.values() if isinstance(v.tensor, list)}
+    if len(list_axes) > 1:
+        raise ValueError("Neural3DBench supports one outer list batch axis per call")
+    list_axis = next(iter(list_axes), None)
+    if list_axis is not None and (not batch or batch[0] != list_axis):
+        raise ValueError("The list axis must be the first batch axis")
+    count = next(len(v.tensor) for v in values.values() if isinstance(v.tensor, list)) if list_axis else 1
+    groups = []
+    for index in range(count):
+        selected = {}
+        for q, value in values.items():
+            axes, tensor = value.axes, value.tensor
+            if list_axis in axes:
+                position = axes.index(list_axis)
+                tensor = tensor[index] if isinstance(tensor, list) else tensor.select(position, index)
+                axes = tuple(a for a in axes if a != list_axis)
+            selected[q] = (tensor, axes)
+        dense_axes = tuple(a for a in batch if a != list_axis)
+        sizes = {a: t.shape[axes.index(a)] for t, axes in selected.values() for a in axes}
+        shape = tuple(sizes[a] for a in dense_axes)
+        if any(n == 0 for n in shape):
+            raise ValueError("Batch axes must be nonempty")
+        raw = {}
+        for role, q in model.input_quantities.items():
+            tensor, axes = selected[q]
+            wanted = specs[role][0]
+            if any(a not in axes for a in wanted):
+                raise ValueError(f"Missing core axes for {role!r}: {wanted}")
+            if set(axes) - set(dense_axes) != set(wanted):
+                raise ValueError(f"Unexpected core axes for {role!r}")
+            present = tuple(a for a in dense_axes if a in axes)
+            order = present + wanted
+            permutation = tuple(axes.index(a) for a in order) + tuple(range(len(axes), tensor.ndim))
+            tensor = tensor.permute(permutation)
+            for dim, axis in enumerate(dense_axes):
+                if axis not in present:
+                    tensor = tensor.unsqueeze(dim)
+            spatial_shape = tuple(sizes[a] for a in wanted)
+            tensor = tensor.expand(shape + spatial_shape + q.shape)
+            raw[role] = tensor.reshape((prod(shape),) + spatial_shape + (specs[role][1],))
+        groups.append((shape, raw))
+    return batch, list_axis, groups
+
+
+def _model_input(model, raw):
+    from .types import ModelInput
+    raw = dict(raw)
+    if model.config.name in _GRIDS:
+        raw["voxel_fields"] = raw["voxel_fields"].movedim(-1, 1)
+    if "point_mask" in raw:
+        raw["point_mask"] = raw["point_mask"].squeeze(-1)
+    return ModelInput(**raw)
+
+
+def validate_contract(model, values):
+    """Run data-dependent input checks before entering torch.func.vmap."""
+    if set(values) != set(model.inputs):
+        raise ValueError(f"Expected input quantities: {[q.name for q in model.inputs]}")
+    sizes = {}
+    for q, v in values.items():
+        _check_value(q, v, sizes)
+    _, _, groups = _prepared_batches(model, values)
+    for _, raw in groups:
+        if model.config.name in _GRAPHS:
+            for i in range(raw["node_features"].shape[0]):
+                model.validate_inputs(_graph_input(model, {r: t[i] for r, t in raw.items()}))
+        else:
+            model.validate_inputs(_model_input(model, raw))
+
+
+def _outputs(model, result, batch_shape):
+    """Restore intrinsic Quantity shapes, never folding sample axes into them."""
+    _, specs, _ = _specs(model)
+    converted = {}
+    for role, q in model.output_quantities.items():
+        tensor = getattr(result, role)
+        if model.config.name in _GRIDS or (model.config.name == "gino" and role == "latent"):
+            tensor = tensor.movedim(1, -1)
+        core_shape = tensor.shape[1:-1]
+        converted[q] = tensor.reshape(batch_shape + core_shape + q.shape)
+    return converted
+
+
+def _graph_input(model, raw):
+    from .types import ModelInput
+    raw = dict(raw)
+    edges = raw["edge_index"]
+    if edges.dtype not in (torch.int32, torch.int64):
+        raise TypeError("edge_index must have an integer dtype")
+    raw["edge_index"] = edges.transpose(-2, -1).long()
+    return ModelInput(**raw)
+
+
+def _evaluate_graphs(model, values):
+    """Pack disjoint meshes once, then split without padding or cross edges."""
+    batch, list_axis, groups = _prepared_batches(model, values)
+    members = [({r: t[i] for r, t in raw.items()}, group_index)
+               for group_index, (_, raw) in enumerate(groups)
+               for i in range(raw["node_features"].shape[0])]
+    nodes, edges, features, lengths = [], [], [], []
+    offset = 0
+    for raw, _ in members:
+        lengths.append(raw["node_features"].shape[0])
+        nodes.append(raw["node_features"])
+        edges.append(raw["edge_index"] + offset)
+        if "edge_features" in raw:
+            features.append(raw["edge_features"])
+        offset += lengths[-1]
+    packed = {"node_features": torch.cat(nodes), "edge_index": torch.cat(edges)}
+    if features:
+        packed["edge_features"] = torch.cat(features)
+    prediction = model._predict(_graph_input(model, packed))
+    outputs = {}
+    for role, q in model.output_quantities.items():
+        split = getattr(prediction, role).split(lengths)
+        collected, cursor = [], 0
+        for shape, raw in groups:
+            count = raw["node_features"].shape[0]
+            tensors = split[cursor:cursor + count]
+            cursor += count
+            collected.append(torch.stack(tensors).reshape(shape + (tensors[0].shape[0],) + q.shape))
+        outputs[q] = Value(collected if list_axis else collected[0], batch + ("vertex",))
+    return outputs
+
+
+def evaluate_contract(model, values):
+    """Run a core call, or own the whole batch for BatchNorm and mesh packing."""
+    if model.config.name in _GRAPHS:
+        return _evaluate_graphs(model, values)
+    if not model.signature.has_ellipsis:
+        specs, _, _ = _specs(model)
+        raw = {role: values[q].reshape((1,) + values[q].shape[:len(specs[role][0])] + (specs[role][1],))
+               for role, q in model.input_quantities.items()}
+        return _outputs(model, model._predict(_model_input(model, raw)), ())
+
+    batch, list_axis, groups = _prepared_batches(model, values)
+    shapes = {tuple(t.shape[1:] for t in raw.values()) for _, raw in groups}
+    if len(shapes) > 1 and uses_batch_statistics(model):
+        raise ValueError("Ragged grid execution requires network-specific handling before BatchNorm")
+    if len(shapes) == 1:
+        # Equal-sized list members participate in one real BatchNorm batch too.
+        raw = {r: torch.cat([g[r] for _, g in groups], dim=0) for r in groups[0][1]}
+        result = model._predict(_model_input(model, raw))
+        flat = _outputs(model, result, (sum(prod(shape) for shape, _ in groups),))
+        outputs = {q: [] for q in model.outputs}
+        cursor = 0
+        for shape, _ in groups:
+            count = prod(shape)
+            for q, tensor in flat.items():
+                outputs[q].append(tensor[cursor:cursor + count].reshape(shape + tensor.shape[1:]))
+            cursor += count
+    else:
+        # Evaluation with frozen statistics needs no common point/grid size.
+        outputs = {q: [] for q in model.outputs}
+        for shape, raw in groups:
+            for q, tensor in _outputs(model, model._predict(_model_input(model, raw)), shape).items():
+                outputs[q].append(tensor)
+    specs = _specs(model)[1]
+    return {q: Value(outputs[q] if list_axis else outputs[q][0], batch + specs[r][0])
+            for r, q in model.output_quantities.items()}
```

</details>

<details>
<summary>normalization.py — فایل افزوده</summary>

```diff
--- /dev/null
+++ src/pinn_engine/networks/normalization.py
@@ -0,0 +1,174 @@
+"""Normalization of dense, ragged, and explicitly masked feature batches.
+
+Inputs use PyTorch's channel-first layout: (B, C, *spatial). A list contains
+pieces of ONE batch, not independent BatchNorm calls. Each real spatial sample
+has equal weight. Dense calls without a mask retain PyTorch's implementation.
+"""
+
+from contextlib import contextmanager
+from contextvars import ContextVar
+
+import torch
+from torch import nn
+
+
+_mask = ContextVar("normalization_mask", default=None)
+
+
+@contextmanager
+def normalization_mask(mask, *, enabled=True):
+    """Scope a spatial mask without threading it through every pointwise layer.
+
+    Dense post-pooling (B, C) layers deliberately do not inherit a spatial mask.
+    Disabled scopes preserve the caller's mask. ContextVars isolate concurrent
+    calls; no module hooks, monkey patches, or persistent per-batch state exist.
+    """
+    if not enabled:
+        yield
+        return
+    token = _mask.set(mask)
+    try:
+        yield
+    finally:
+        _mask.reset(token)
+
+
+def has_normalization_mask():
+    return _mask.get() is not None
+
+
+def batch_norms(module):
+    """Inspect actual layers rather than configuration spellings."""
+    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
+
+
+def uses_batch_statistics(module):
+    return any(m.training or (m.running_mean is None and m.running_var is None)
+               for m in batch_norms(module))
+
+
+def masked_point_inputs(points, features, mask):
+    """Keep invalid inputs out of pointwise convolutions and their gradients."""
+    if mask is None:
+        return points, features
+    valid = mask.unsqueeze(-1)
+    return (torch.where(valid, points, 0.),
+            None if features is None else torch.where(valid, features, 0.))
+
+
+def _checked_mask(tensor, mask):
+    if mask is None:
+        return None
+    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
+        raise TypeError("Normalization masks must be boolean Tensors")
+    if mask.shape != tensor.shape[:1] + tensor.shape[2:]:
+        raise ValueError("Mask shape must be (B, *spatial), without the channel axis")
+    if mask.device != tensor.device:
+        raise ValueError("Normalization mask and features must be on the same device")
+    return mask
+
+
+class _BatchNorm:
+    def forward(self, inputs, mask=None):
+        if isinstance(inputs, torch.Tensor):
+            # The ambient mask belongs to point/spatial layers, not pooled FCs.
+            if mask is None and inputs.ndim > 2:
+                mask = _mask.get()
+            if mask is None:
+                return super().forward(inputs)
+            return self.forward([inputs], [mask])[0]
+        if not isinstance(inputs, list) or not inputs:
+            raise TypeError("BatchNorm expects a Tensor or a nonempty flat Tensor list")
+        masks = [None] * len(inputs) if mask is None else mask
+        if not isinstance(masks, list) or len(masks) != len(inputs):
+            raise ValueError("List input requires a matching list of masks")
+        rows, valid_masks, lengths = [], [], []
+        first = inputs[0]
+        for tensor, valid in zip(inputs, masks):
+            if not isinstance(tensor, torch.Tensor):
+                raise TypeError("BatchNorm list members must be Tensors")
+            self._check_input_dim(tensor)
+            if (tensor.ndim != first.ndim or tensor.shape[1] != first.shape[1]
+                    or tensor.dtype != first.dtype or tensor.device != first.device):
+                raise ValueError("BatchNorm members must share rank, channels, dtype, and device")
+            valid = _checked_mask(tensor, valid)
+            flat = tensor.movedim(1, -1).reshape(-1, tensor.shape[1])
+            selected = flat if valid is None else flat[valid.reshape(-1)]
+            rows.append(selected)
+            valid_masks.append(valid)
+            lengths.append(selected.shape[0])
+        if not sum(lengths):
+            raise ValueError("BatchNorm needs at least one real sample")
+        # Native BatchNorm owns affine parameters, momentum, biased training
+        # variance, unbiased running variance, and ONE buffer update per call.
+        packed = torch.cat(rows).T.reshape(1, first.shape[1], -1)
+        if first.ndim == 2:
+            packed = packed.squeeze(0).T
+        else:
+            packed = packed.reshape(packed.shape + (1,) * (first.ndim - 3))
+        normalized = super().forward(packed).movedim(1, -1).reshape(-1, first.shape[1])
+        outputs = []
+        for tensor, valid, values in zip(inputs, valid_masks, normalized.split(lengths)):
+            shape = tensor.shape[:1] + tensor.shape[2:] + tensor.shape[1:2]
+            if valid is not None:
+                restored = values.new_zeros((valid.numel(), tensor.shape[1]))
+                restored[valid.reshape(-1)] = values
+                values = restored
+            outputs.append(values.reshape(shape).movedim(-1, 1))
+        return outputs
+
+
+class BatchNorm1d(_BatchNorm, nn.BatchNorm1d):
+    """BatchNorm1d with joint list statistics and optional spatial masks."""
+
+
+class BatchNorm2d(_BatchNorm, nn.BatchNorm2d):
+    """BatchNorm2d with joint list statistics and optional spatial masks."""
+
+
+class BatchNorm3d(_BatchNorm, nn.BatchNorm3d):
+    """BatchNorm3d with joint list statistics and optional spatial masks."""
+
+
+class InstanceNorm1d(nn.InstanceNorm1d):
+    """Per-case, per-channel point normalization; supports masks and lists."""
+
+    def forward(self, inputs, mask=None):
+        if isinstance(inputs, list):
+            if not inputs:
+                raise ValueError("InstanceNorm needs a nonempty Tensor list")
+            masks = [None] * len(inputs) if mask is None else mask
+            if not isinstance(masks, list) or len(masks) != len(inputs):
+                raise ValueError("List input requires a matching list of masks")
+            return [self(t, m) for t, m in zip(inputs, masks)]
+        if mask is None and inputs.ndim == 3:
+            mask = _mask.get()
+        if mask is None:
+            return super().forward(inputs)
+        self._check_input_dim(inputs)
+        mask = _checked_mask(inputs, mask)
+        if self.track_running_stats:
+            raise ValueError("Masked InstanceNorm requires track_running_stats=False")
+        # Reductions keep their dimensions, so this also works under vmap.
+        valid = mask.unsqueeze(1)
+        work = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
+        clean = torch.where(valid, work, 0.)
+        count = valid.sum(-1, keepdim=True).clamp_min(1)
+        mean = clean.sum(-1, keepdim=True) / count
+        centered = torch.where(valid, work - mean, 0.)
+        variance = centered.square().sum(-1, keepdim=True) / count
+        output = centered * torch.rsqrt(variance + self.eps)
+        if self.affine:
+            output = output * self.weight[None, :, None] + self.bias[None, :, None]
+        return output.masked_fill(~valid, 0.).to(inputs.dtype)
+
+
+def point_normalization(kind, channels, *, pooled=False):
+    """PointNet policy: instance normalization ends when point pooling ends."""
+    if kind == "batch":
+        return BatchNorm1d(channels)
+    if kind == "instance" and not pooled:
+        return InstanceNorm1d(channels, affine=True, track_running_stats=False)
+    if kind in {"none", "instance"}:
+        return nn.Identity()
+    raise ValueError("normalization must be batch, instance, or none")
```

</details>
