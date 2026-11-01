# شبکه‌های Neural3DBench در قرارداد موتور

هر ۱۱ شبکه مستقیماً از `Model` ارث می‌برند و `signature` و `evaluate` خودشان را دارند. `BaseNetwork`، `BaseGraphNetwork` و متد `predict` حذف شده‌اند؛ اجرای معمول همان `network(values)` است. پارامترها و زیرماژول‌های اصلی حفظ شده‌اند:

```python
import torch
from pinncore import Quantity, Solver, Value
from pinncore.networks import SIREN, SIRENConfig

xyz = Quantity("wing_coordinates", (3,))
stress = Quantity("wing_stress", (3, 3))
network = SIREN(
    SIRENConfig(output_dim=9, hidden_dim=32),
    inputs={"coordinates": xyz},
    outputs={"predictions": stress},
)
solver = Solver([network], outputs=stress)
values = {xyz: Value(torch.rand(4, 64, 3), ("scenario", "point"))}
result = solver(values)[stress]  # Tensor (4, 64, 3, 3), same sample axes.
result.tensor.square().mean().backward()
```

کلید نگاشت‌ها نقش موجود در شبکه است؛ خود Quantity و نامش را کاربر انتخاب می‌کند. `inputs` و `outputs` آرگومان‌های اجباری init هستند. ورودی اختیاری فقط با حضور در همین نگاشت فعال و هنگام اجرا الزامی می‌شود. تعداد مؤلفه‌ها باید با config برابر باشد؛ مثلاً ۹ خروجی به `(3,3)` و یک خروجی به `()` تبدیل می‌شود، بدون تغییر محورهای نمونه. نگاشت نامعتبر، خروجی تکراری و ورودی استفاده‌نشده خطا دارند. اگر config ویژگی اضافه بخواهد، `point_features` هم اجباری است.

در جدول، محورها مربوط به یک فراخوانی core هستند. ابعاد ذاتی Quantity جدا و انتهایی‌اند. برای نمونه `coordinates` با شکل ذاتی `(3,)` و محور `point` مقدار `(N,3)` دارد.

| شبکه | ورودی‌های پشتیبانی‌شده | predictions | latent |
|---|---|---|---|
| GCN | `node_features`, `edge_index` | `vertex`, مؤلفه `output_dim` | ندارد |
| GAT | `node_features`, `edge_index` | `vertex`, مؤلفه `output_dim` | `vertex`, مؤلفه `hidden_dim` |
| MeshGraphNet | `node_features`, `edge_index`, `edge_features` | `vertex`, مؤلفه `output_dim` | `vertex`, مؤلفه `latent_dim` |
| PointNet / PointNet2 | `coordinates`, ویژگی اضافه در صورت نیاز config، `point_mask` اختیاری | `point` در segmentation؛ بدون محور مکانی در classification | embedding بدون محور مکانی، مؤلفه `global_dim` |
| Transolver | `coordinates`, ویژگی اضافه در صورت نیاز config، `point_mask` اختیاری | `point`, مؤلفه `output_dim` | `point`, مؤلفه `hidden_dim` |
| SIREN | `coordinates`, ویژگی اضافه در صورت نیاز config، `point_mask` اختیاری | `point`, مؤلفه `output_dim` | `point`, مؤلفه `hidden_dim` |
| GINO | `coordinates`, `point_features`، `point_mask` و `query_coordinates` اختیاری | `point`؛ با query جدا، محور `query` | `latent_z,latent_y,latent_x`، مؤلفه `width` |
| FNO3D | `voxel_fields` | `z,y,x`، مؤلفه `output_channels` | `z,y,x`، مؤلفه `width` |
| UNet3D | `voxel_fields` | `z,y,x`، مؤلفه `output_channels` | `bottleneck_z,bottleneck_y,bottleneck_x`، مؤلفه `base_channels * channel_multiplier ** (levels-1)` |
| DeepNURBS | `coordinates` در فضای پارامتری دوبعدی | `point`, مؤلفه `output_dim` | میدان عصبی قبل از اعمال قید، `point`, مؤلفه `output_dim` |

در ورودی گراف، `edge_index` یک Quantity با دو مؤلفه و مقدار `(E,2)` با dtype صحیح است؛ خود مدل آن را برای الگوریتم اصلی به `(2,E)` می‌برد. ورودی voxel هم در موتور channel-last است: `(Z,Y,X,C)`، یا `(Z,Y,X)` برای Quantity اسکالر. `point_mask` بولی است. GINO همچنان مختصات نرمال‌شده در `[0,1]` می‌خواهد. هندسهٔ DeepNURBS همچنان ثابت و داخل config است.

## محورهای batch و استقلال

- اگر **خود ماژول‌ها واقعاً BatchNorm داشته باشند**، Signature همهٔ ورودی‌ها و خروجی‌ها را با پیشوند `...` معرفی می‌کند. همهٔ محورهای batch با نام تطبیق و به یک B تخت تبدیل می‌شوند؛ BatchNorm یک‌بار روی batch واقعی اجرا می‌شود. بررسی تنها بر اساس رشتهٔ config نیست؛ مثلاً GCN تک‌لایه normalization نهایی ندارد.
- شبکه‌های گرافی حتی بدون BatchNorm همین پیشوند را دارند. همهٔ گراف‌ها با جابه‌جایی اندیس یال‌ها به یک گراف جدا از هم تبدیل و با یک اجرای شبکه پردازش می‌شوند؛ بعد خروجی شکسته می‌شود. ورودی‌های مشترک مثل connectivity بدون محور سناریو روی محورهای batch به اشتراک گذاشته می‌شوند.
- بقیهٔ شبکه‌ها `...` ندارند. موتور روی محورهای اضافی `vmap` و روی محور بیرونی لیستی حلقه اجرا می‌کند. Dropout برای نمونه‌های مستقل تصادفی‌های جدا می‌سازد؛ پیش‌فرض مدل‌های عمومی موتور همچنان خطا روی تصادف داخل vmap است.
- در مسیر Model و Solver، PointNet و PointNet++ که واقعاً BatchNorm دارند فقط ورودی Tensor می‌پذیرند؛ وجود هر ورودی لیستی، حتی با اعضای هم‌اندازه یا در eval، خطاست. padding و تبدیل خودکار لیست به batch برای این دو شبکه انجام نمی‌شود؛ ماسک صریح همراه Tensor همچنان پشتیبانی می‌شود. مدل‌های بدون BatchNorm لیست را طبق مسیر معمول موتور پردازش می‌کنند و خروجی لیستی را لیست نگه می‌دارند. گراف‌ها همچنان بدون padding بسته‌بندی می‌شوند. اجرای کامل FNO/U-Net با gridهای ragged و آمار batch هنوز adapter اختصاصی می‌خواهد. قابلیت لیست در خود لایهٔ مستقل BatchNorm حفظ شده است. جزئیات در [نرمال‌سازی مشترک](normalization.fa.md) است.

استقلال `point` برای SIREN و DeepNURBS برابر True است؛ برای PointNet، PointNet2، Transolver و خروجی پیش‌فرض GINO برابر False. queryهای جدا در GINO از یکدیگر مستقل‌اند، در حالی که encoder همچنان همهٔ نقاط منبع را ترکیب می‌کند. محور `vertex` شبکه‌های گراف و محورهای مکانی FNO/U-Net وابسته‌اند. محورهای اضافی batch مستقل‌اند مگر BatchNorm از آمار batch استفاده کند؛ در eval با آمار ثابت دوباره مستقل‌اند. Signature هنگام اجرا وضعیت ماژول‌های BatchNorm را می‌خواند.

`independence=False` مانع آموزش عادی و `loss.backward()` نیست؛ backend AD فعلی طبق قرارداد روی درخواست مشتق با محور مشترکِ وابسته خطا می‌دهد. مشتق تخصصی شبکه‌ها اضافه نشده است.

## سازگاری و محدودیت‌های فعلی

`model(values)` و `Solver` نگاشت Quantity به Value می‌گیرند و فقط خروجی‌های عددی انتخاب‌شده را می‌دهند. محاسبات هر شبکه داخل `evaluate` همان کلاس است؛ `prepare_inputs` و `finish_outputs` فقط layout، batch و packing گراف را مدیریت می‌کنند. بررسی وابسته به مقدار با `check` داخل evaluate انجام می‌شود؛ بررسی شکل و dtype با شرط معمولی است.

کلاس‌های قدیمی `ModelInput`، `ModelOutput`، `ModelCapabilities` و representation adapterهای بخش networks حذف شده‌اند. ورودی/خروجی شبکه‌ها فقط نگاشت Quantity/Value است؛ تست‌ها نیز از دیکشنری ساده استفاده می‌کنند. `geocore.ModelInput` کلاس مستقلی است و در خروجی GeoCore باقی مانده است. کد auxiliary و metadata الگوریتم‌های اصلی حفظ شده، ولی خروجی عمومی موتور نیست. انتخاب فقط latent نیز لزوماً محاسبهٔ head پیش‌بینی را حذف نمی‌کند.

برای امکان vmap، GINO از ماسک با طول ثابت به جای boolean indexing استفاده می‌کند. در مسیر Model، checkpoint داخلی kernel خاموش است ولی موتور **کل تابع batch‌شده را بیرون تمام vmapها** با `use_reentrant=False` checkpoint می‌کند. batching خودکار حفظ می‌شود و نتایج میانی کمتری برای backward ذخیره می‌شوند؛ در عوض، هنگام مشتق‌گیری محاسبات دوباره اجرا می‌شوند و ناحیهٔ replay بزرگ‌تر از chunkهای kernel قبلی است. محدودیت workspace زوج‌نقاط و chunkهای kernel حفظ شده است. کاهش حافظهٔ ذخیره‌شده برای backward به معنی تضمین مقدار مشخصی از کاهش peak حافظه یا سرعت نیست.

این رفتار فقط برای GINO با `checkpoint_batches=True` پیش‌فرض فعال است؛ برای مقایسه می‌توان روی instance مقدار False گذاشت. در مدل‌های دیگر پیش‌فرض False است. این گزینه به مسیر عادی batching موتور مربوط است؛ مسیر wildcard مدل، batching را خودش مدیریت می‌کند. در batch لیستی برای هر عضو checkpoint جدا و در حالت بدون محور batch نیز یک checkpoint روی فراخوانی core ساخته می‌شود. در `torch.no_grad()` موتور checkpoint را کنار می‌گذارد. مسیر جداگانهٔ predict وجود ندارد و checkpoint داخلی kernel در اجرای شبکه خاموش است.

checkpoint نیازمند محاسبهٔ قابل replay است؛ نباید آن را بدون بررسی برای مدل‌های دارای تغییر state مانند آمار BatchNorm فعال کرد. خروجی، گرادیان‌ها و مشتق‌های تا مرتبهٔ سوم GINO، از جمله مشتق‌های query در Solver و آموزش با loss شامل مشتق مرتبهٔ سوم، با مسیر بدون checkpoint تست شده‌اند. پشتیبانی همهٔ ترکیب‌های torch.func یا compile از این تست‌ها نتیجه نمی‌شود.

کد شبکه‌ها در `src/pinncore/networks` قرار دارد و با نام `pinncore.networks` نصب می‌شود؛ نصب پکیج عمومی `models` یا تغییر `sys.path` لازم نیست. مثال‌ها/تست‌های قدیمی upstream که constructor بدون inputs/outputs و فراخوانی مستقیم با ModelInput دارند، API قدیمی را نشان می‌دهند؛ مسیر جدید نمونهٔ بالاست.

تست‌ها روی CPU اجرا شده‌اند. پشتیبانی عملی همهٔ شبکه‌ها از GPU، fullgraph compilation و سرعت بیشتر تضمین نشده است. اعتبارسنجی داده، بسته‌بندی لیست و برخی الگوریتم‌ها حلقه/شرط دارند. افزودن domain و کنترل برابر بودن domain ورودی/خروجی مکانی مطابق یادداشت طراحی هنوز برای آینده است.

گزینهٔ normalization="instance" در PointNet فقط روی point هر هندسه اعمال می‌شود و بعد از pooling normalization ندارد؛ از batch هندسه‌های دیگر مستقل است. تغییرات دقیق نسبت به منبع در [گزارش تطبیق شبکه‌ها](network_upstream_changes.fa.md) ثبت شده‌اند.
