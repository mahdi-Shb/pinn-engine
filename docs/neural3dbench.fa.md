# شبکه‌های Neural3DBench در قرارداد موتور

هر ۱۱ شبکه مستقیماً subclass از `Model` هستند و پارامترها/زیرماژول‌های اصلی را نگه می‌دارند؛ adapter جدا لازم نیست. همراه خود موتور نصب می‌شوند:

```python
import torch
from pinn_engine import Quantity, Solver, Value
from pinn_engine.neural3dbench import SIREN, SIRENConfig

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
- لیست خروجی لیست می‌ماند. هیچ padding خودکاری نداریم. BatchNorm با آمار batch و اندازه‌های core متفاوت خطای روشن می‌دهد؛ در eval با آمار ثابت، لیست ragged قابل اجراست. ماسک صریحِ ورودی هم BatchNorm اصلی را به masked BatchNorm تبدیل نمی‌کند؛ آمار آن همچنان از slotهای ماسک‌شده تأثیر می‌گیرد.

استقلال `point` برای SIREN و DeepNURBS برابر True است؛ برای PointNet، PointNet2، Transolver و خروجی پیش‌فرض GINO برابر False. queryهای جدا در GINO از یکدیگر مستقل‌اند، در حالی که encoder همچنان همهٔ نقاط منبع را ترکیب می‌کند. محور `vertex` شبکه‌های گراف و محورهای مکانی FNO/U-Net وابسته‌اند. محورهای اضافی batch مستقل‌اند مگر BatchNorm از آمار batch استفاده کند؛ در eval با آمار ثابت دوباره مستقل‌اند. Signature هنگام اجرا وضعیت ماژول‌های BatchNorm را می‌خواند.

`independence=False` مانع آموزش عادی و `loss.backward()` نیست؛ backend AD فعلی طبق قرارداد روی درخواست مشتق با محور مشترکِ وابسته خطا می‌دهد. مشتق تخصصی شبکه‌ها اضافه نشده است.

## سازگاری و محدودیت‌های فعلی

`model(values)` و `Solver` فقط خروجی‌های انتخاب‌شدهٔ عددی را تحویل می‌دهند. `model.predict(ModelInput(...))` مسیر عددی با شکل‌های اصلی Neural3DBench است و `ModelOutput`، auxiliary و metadata را حفظ می‌کند. کد آن خروجی‌ها حذف نشده است. الگوریتم اصلی ممکن است head خروجی انتخاب‌نشده را هم اجرا کند؛ حذف محاسبهٔ headها در این تغییر پیاده نشده است.

برای امکان vmap، GINO در مسیر Model از ماسک با طول ثابت به جای boolean indexing استفاده می‌کند و activation checkpoint را خاموش می‌کند؛ این مسیر ممکن است حافظهٔ آموزش بیشتری بخواهد. در `predict(ModelInput)` checkpoint موجود همچنان فعال است. محدودیت workspace زوج‌نقاط و chunkهای kernel حفظ شده است.

کد شبکه‌ها در `third_party/neural3dbench/models` نگه داشته شده ولی با نام اختصاصی `pinn_engine.neural3dbench` بسته‌بندی می‌شود؛ نصب پکیج عمومی `models` یا تغییر `sys.path` لازم نیست. مثال‌ها/تست‌های قدیمی upstream که constructor بدون inputs/outputs و فراخوانی مستقیم با ModelInput دارند، API قدیمی را نشان می‌دهند؛ مسیر جدید نمونهٔ بالاست.

تست‌ها روی CPU اجرا شده‌اند. پشتیبانی عملی همهٔ شبکه‌ها از GPU، fullgraph compilation و سرعت بیشتر تضمین نشده است. اعتبارسنجی داده، بسته‌بندی لیست و برخی الگوریتم‌ها حلقه/شرط دارند. افزودن domain و کنترل برابر بودن domain ورودی/خروجی مکانی مطابق یادداشت طراحی هنوز برای آینده است.
