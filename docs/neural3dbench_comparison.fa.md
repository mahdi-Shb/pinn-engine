# تفاوت Neural3DBench اصلی با شبکه‌های فعلی PINNCore

> پاک‌سازی بعدی: ModelInput/Output/Capabilities و representation adapterهای بخش networks حذف شده‌اند. اشاره به نگهداری این کلاس‌ها در متن/diff زیر مربوط به snapshot پیشین است؛ geocore.ModelInput کلاس دیگری است و حذف نشده است.

> نام جاری پکیج `pinncore` و مسیر آن `src/pinncore` است. نام uppercase در متن و diff این گزارش به snapshot کد e256e93 پیش از تغییر حروف اشاره دارد؛ API جاری `from pinncore import ...` است.

تاریخ بازبینی: ۲۰۲۶-۱۱-۰۱. کد مقایسه‌شده: commit موتور `e256e93`، پس از حذف BaseNetwork. این گزارش وضعیت فعلی را توضیح می‌دهد؛ diff آن نیز از نو تولید شده است.

## خلاصهٔ قابل ارسال به همکار

> معماری‌ها و الگوریتم‌های اصلی از Neural3DBench آمده‌اند. هر شبکه اکنون مستقیماً یک Model موتور است و signature و evaluate خودش را دارد. BaseNetwork و BaseGraphNetwork و رابط predict حذف شده‌اند. اتصال Quantity/Value، batching، بسته‌بندی گراف، check و normalization ماسک‌پذیر قابلیت‌های موتورند. فرمول‌های اصلی عمدتاً حفظ شده‌اند؛ API، آمار ورودی ماسک‌دار و نحوهٔ اجرای batch/checkpoint تغییر واقعی دارند.

## ۱. مبنای دقیق مقایسه

- منبع: [n94kholdi/neural3dbench](https://github.com/n94kholdi/neural3dbench)، commit `1b635f4d1c3b47fab5c9e9a46b1824b789e478ee`.
- HEAD و main منبع در این بازبینی دوباره با `git ls-remote` بررسی شدند و هنوز همان commit بودند؛ دربارهٔ branchهای دیگر ادعایی نداریم.
- snapshot متناظر منبع در تاریخچهٔ موتور: `80e5c8d:third_party/neural3dbench/models`.
- محل کد فعلی: `src/PINNCore/networks`؛ import عمومی: `PINNCore.networks`. نام repo و پوشهٔ checkout همچنان `pinn_engine` است؛ نام پکیج نصب‌شدنی `PINNCore` است.
- از ۴۳ فایل Python منبع، ۱۳ فایل متنی یکسان، ۲۸ فایل تغییرکرده و ۲ فایل حذف‌شده‌اند. پایان خط Windows/Linux تغییر محتوایی شمرده نشده است.
- دو فایل داخل networks افزوده شده‌اند: `_contract.py` و `normalization.py`. تابع عمومی check نیز در `src/PINNCore/checks.py` افزوده شده است.
- فایل‌های مرجع خارج از models، از جمله مثال‌ها، تست‌ها، mesh adaptation و rollout، حفظ شده‌اند؛ نگهداری آن‌ها به معنی تبدیلشان به Model موتور نیست. متن `models/ModelInput.md` در `docs/network_input.md` و مجوز منبع همراه شبکه‌ها حفظ شده است.

«دلیل تغییر ما» در ادامه به تصمیم موتور اشاره دارد؛ نیت همکار دربارهٔ ساختار قبلی حدس زده نشده است.

## ۲. رابط و ساختار کلاس‌ها

| موضوع | منبع اصلی | نسخهٔ فعلی |
|---|---|---|
| کلاس پایه | BaseNetwork از nn.Module؛ شبکه‌های گرافی از BaseGraphNetwork | هر ۱۱ کلاس مستقیماً از Model ارث می‌برند؛ Model خود nn.Module است. هر دو پایهٔ قبلی حذف شده‌اند. |
| سازنده | config اختیاری | config و نگاشت‌های keyword-only اجباری inputs و outputs به Quantity؛ پارامترها در همان زیرماژول‌های اصلی‌اند. |
| فراخوانی | network(ModelInput) و predict؛ خروجی ModelOutput | network({Quantity: Value})؛ خروجی نگاشت Quantity به Value، قابل استفاده در Solver. |
| محاسبات | forward اختصاصی شبکه | evaluate اختصاصی همان کلاس؛ forward استاندارد Model محورهای اضافی را مدیریت می‌کند. predict و _predict وجود ندارند. |
| قرارداد | required_inputs، capabilities و supported_representations | signature و نگاشت نقش‌ها؛ خصوصیات و validate_inputs قدیمی از کلاس‌های شبکه حذف شده‌اند. |
| validation | پیش از pipeline عددی | داخل evaluate و helperهای فراخوانده‌شده از آن؛ شرط مقدار با check، شرط شکل/dtype با if معمولی. |
| دادهٔ میانی | ModelInput و ModelOutput | دیکشنری نقش به Tensor؛ helperهای مشترک فقط layout و packing را تبدیل می‌کنند. |

ModelInput، ModelOutput، ModelCapabilities و representation adapterهای قدیمی هنوز بدون بازنویسی و با export عمومی برای داده/مرجع باقی‌اند؛ **رابط اجرای شبکه نیستند**. کد تولید auxiliary و metadata نیز داخل الگوریتم‌ها حفظ شده، ولی خروجی عمومی قابل انتخاب موتور نیست. در نسخهٔ جاری مسیر جداگانه‌ای برای دریافت آن‌ها از predict نداریم.

تمام فراخوانی‌های عمومی شبکه از __call__/forward معمول nn.Module می‌گذرند؛ توضیح قدیمی دربارهٔ bypass شدن hookها در predict دیگر رفتار جاری نیست. سازگاری بارگذاری state_dict با config قبلی حفظ شده، اما pickle کامل با مسیر قدیمی `models.*` یا `pinn_engine.*` تضمین نشده است. مثال‌های قدیمی third_party نیز باید برای API جدید بازنویسی شوند.

## ۳. نقش Quantity، شکل و انتخاب ورودی/خروجی

```python
import torch
from PINNCore import Quantity, Value
from PINNCore.networks import SIREN, SIRENConfig

xyz = Quantity("wing_coordinates", (3,))
stress = Quantity("wing_stress", (3, 3))
network = SIREN(
    SIRENConfig(output_dim=9, hidden_dim=16),
    inputs={"coordinates": xyz},
    outputs={"predictions": stress},
)
result = network({xyz: Value(torch.rand(2, 8, 3), ("scenario", "point"))})
# result[stress].tensor.shape == (2, 8, 3, 3)
```

- کلیدهای نقش مانند coordinates و predictions مشخص‌اند؛ نام و هویت Quantity را کاربر انتخاب می‌کند.
- تعداد مؤلفه‌ها با config بررسی می‌شود: مثلاً ۹ کانال به (3,3) و یک کانال به اسکالر () reshape می‌شود؛ محورهای نمونه جزو شکل ذاتی نیستند.
- نقش ناشناخته، ورودی اجباریِ غایب، خروجی خالی یا Quantity تکراری/ناسازگار خطا دارد. ورودی و خروجی یک binding نیز باید Quantityهای متفاوت باشند.
- نقش اختیاری مانند point_mask یا query_coordinates در init انتخاب می‌شود؛ پس از انتخاب، هنگام اجرا اجباری است. features غیرصفرِ config نیز point_features را اجباری می‌کند.
- فقط predictions و latent موجود قابل انتخاب‌اند؛ GCN latent ندارد. تمام فیلدهای dataclass قدیمی ModelInput به نقش موتور تبدیل نشده‌اند.
- انتخاب فقط latent یا predictions، فعلاً headها یا auxiliaryهای انتخاب‌نشده را از محاسبه حذف نمی‌کند.

## ۴. helperهای مشترک؛ چه چیزی هنوز مشترک است؟

| helper در _contract.py | مسئولیت |
|---|---|
| _specs | تعریف نقش، تعداد مؤلفه و محور هر معماری بر اساس config.name. |
| initialize_contract | بررسی bindingها هنگام ساخت؛ شبکه خودش config و ماژول‌های عددی را می‌سازد. |
| contract_signature | ساخت signature با قواعد batch و استقلال متناسب با معماری و BatchNorm واقعی. |
| _prepared_batches | تطبیق/اشتراک محورهای هم‌نام، broadcast محور غایب، تخت‌کردن محورهای dense و گروه‌بندی لیست؛ بدون padding. |
| _native_input و _outputs | جابه‌جایی channel، شکل mask و بازیابی شکل ذاتی Quantity. |
| _pack_graphs | بررسی index محلی هر گراف، اتصال node/edge و offset یال‌ها. |
| prepare_inputs | تحویل requestهای عددی و اطلاعات layout به evaluate؛ الگوریتم شبکه را اجرا نمی‌کند. |
| finish_outputs | بازیابی محورهای dense/list و شکستن خروجی گراف؛ dispatch عددی به مدل ندارد. |

مسیر قبلی validate_contract و آماده‌سازی مضاعف ورودی برای validation پیش از vmap حذف شده است. با این حال، تبدیل layout، اسکن BatchNorm، cat/stack و bookkeeping هنوز هزینه دارند؛ حذف تمام اضافه‌کاری یا pruning با compile ادعا نمی‌شود.

registry اکنون subclass از Model می‌پذیرد و create_model(existing_model) همان نمونه را برمی‌گرداند. factory همچنان سازندهٔ config/inputs/outputs را فرض می‌کند. مدل سفارشی مجبور به استفاده از _specs نیست؛ ولی معماری جدیدی که helperهای فعلی را استفاده کند، باید در سیاست config.name/محورها نیز تعریف شود. قواعد امضا از بدنهٔ دلخواه مدل استنتاج نمی‌شوند.

## ۵. محور، batching و استقلال

| شبکه | محورهای core اصلی | خروجی و رفتار مهم |
|---|---|---|
| PointNet و PointNet++ | point | segmentation روی point؛ classification و latent بدون point. |
| SIREN و DeepNURBS | point | هر نقطه مستقل؛ DeepNURBS مختصات پارامتری دوبعدی می‌گیرد. |
| Transolver | point | attention نقاط را به هم وابسته می‌کند. |
| GINO | point منبع؛ query اختیاری | query جدا از queryهای دیگر مستقل است؛ پیش‌بینی روی مختصات منبع وابسته است. latent روی latent_z/y/x است. |
| FNO3D | z,y,x | convolution طیفی وابسته به نقاط grid؛ latent هم‌اندازهٔ grid. |
| UNet3D | z,y,x | latent محورهای bottleneck_z/y/x جدا دارد، چون resolution کاهش یافته است. |
| GCN، GAT و MeshGraphNet | vertex و edge | خروجی node-wise؛ message passing رأس‌ها را به هم وابسته می‌کند. |

محورهای بیرون core با vmap و محور بیرونی لیستی با حلقهٔ ترتیبی Model پردازش می‌شوند. نام محور تطابق/اشتراک را تعیین می‌کند، نه برابری عددی اندازه‌ها. dropout در vmap با randomness="different" نمونهٔ مستقل می‌گیرد؛ توالی بیت‌به‌بیت RNG با forward batched منبع تضمین نشده است.

وجود هر ماژول واقعی BatchNorm، signature را whole-batch با `...` می‌کند؛ تمام گراف‌ها نیز حتی بدون BN همین مسیر را دارند. شبکه‌های PointNet و PointNet++ دارای BN **هر ورودی لیستی را حتی در eval و با اعضای هم‌اندازه رد می‌کنند**. مدل‌های بدون BN لیست را طبق قرارداد معمول می‌پذیرند. adapter موقت padding خودکار حذف شده و قابلیت جاری نیست.

در FNO/U-Net، لیست‌های هم‌اندازه با BN یک batch مشترک می‌شوند؛ gridهای متفاوت با آمار batch خطا می‌دهند، ولی با running statistics ثابت در eval می‌توانند جدا اجرا شوند. padding عمومی grid افزوده نشده است؛ padding خود معماری FNO یا upsample/pad/crop اصلی U-Net موضوع جداگانه و حفظ‌شده است.

در رابط موتور، voxelها channel-last هستند و قبل از محاسبه به channel-first منبع تبدیل می‌شوند. edge_index در Value به شکل (E,2) است و برای محاسبه به (2,E) و long تبدیل می‌شود. بعد B=1 فراخوانی core داخل helper اضافه/حذف می‌شود؛ ابعاد ذاتی انتهایی باقی می‌مانند.

استقلال point برای SIREN/DeepNURBS True و برای PointNet/PointNet++/Transolver و GINO روی منبع False است. محورهای مکانی grid و vertex گراف نیز وابسته‌اند. محورهای batch با BN هنگام استفاده از آمار batch وابسته و در eval با آمار running ثابت مستقل اعلام می‌شوند. این metadata قراردادی و محافظه‌کارانه است؛ حتی BN ثبت‌شده ولی استفاده‌نشده تشخیص داده می‌شود. تغییر معماری مستلزم بازبینی استقلال است.

AD موتور درخواست مشتق با محور مشترکِ وابسته را طبق قرارداد رد می‌کند؛ backward معمول آموزش برقرار است. Jacobian تخصصی، مشتق FFT/mesh یا تحلیل خودکار dataflow شبکه در این تغییر اضافه نشده است.

## ۶. check و validation داخل evaluate

تابع عمومی check(condition, message) یک Tensor بولی می‌گیرد و اگر هر عضو False باشد ValueError می‌دهد. عملیات سفارشی آن قانون vmap دارد؛ بنابراین evaluate می‌تواند یک نمونه را ببیند ولی شرط همهٔ اعضای batch و batch تودرتو بررسی شود. شکست یک عضو کل فراخوانی را متوقف می‌کند. if پایتونیِ مبتنی بر مقدار Tensor همچنان خودکار vmappable نشده است.

چک ماسک غیرتهی PointNetها، SIREN، Transolver و GINO، مقادیر finite/بازهٔ GINO و DeepNURBS، و indexهای محلی گراف با این ابزار انجام می‌شوند. finite و بازهٔ GINO با where ماسک می‌شوند تا boolean slicing با طول داده‌محور داخل vmap رخ ندهد. check عدد شبکه را تغییر نمی‌دهد؛ مسیر assertion بولی است. آزمایش‌های AD روی مثال‌ها پشتیبانی از مشتق را بررسی می‌کنند؛ تضمین torch.compile یا GPU کامل از آن‌ها نتیجه نمی‌شود. خواندن شرط CUDA نیازمند همگام‌سازی با میزبان است.

## ۷. normalization؛ تغییر رفتاری مستقل از رابط

- BatchNorm1d/2d/3d مشترک subclass کلاس‌های PyTorch هستند؛ Tensor بدون mask مستقیماً از مسیر native می‌گذرد. پارامترها، momentum، affine و running buffers native مانده‌اند. isinstance حفظ می‌شود، ولی type دقیق کلاس عوض شده است.
- خود لایهٔ مستقل BN، لیستِ قطعات یک batch و mask با شکل (B,*spatial) می‌پذیرد: ردیف‌های معتبر جمع، BN یک‌بار اجرا و خروجی دوباره جدا می‌شود. هر نقطه وزن برابر دارد؛ هندسهٔ بزرگ‌تر سهم بیشتری دارد. این قابلیت به معنی پذیرش لیست توسط کل PointNet دارای BN نیست.
- mask صریح در PointNet/PointNet++ اکنون آمار normalization را هم محدود می‌کند؛ در منبع، maskِ pooling/خروجی لزوماً جلوی ورود slot نامعتبر به BN را نمی‌گرفت. این تغییر واقعیِ مقدار خروجی و آمار آموزش است.
- ورودی نامعتبر پیش از Conv صفر می‌شود تا NaN نتواند گرادیان وزن‌ها را آلوده کند. mask با ContextVar فقط در همان فراخوانی منتقل و در خروج/exception به حالت قبل برمی‌گردد؛ cache دائمی batch نیست.
- در PointNet++ ماسک با سطح grouping/propagation منطبق می‌شود. تکرار همسایهٔ واقعی در slotهای اضافی radius grouping مطابق منبع در آمار حفظ شده است؛ در group_all ماسک منبع و در propagation ماسک مقصد اعمال می‌شود.
- گزینهٔ جدید instance در PointNet روی point هر هندسه جدا، affine=True و بدون running stats است. بعد از pooling normalization نداریم و bias FCهای بدون norm فعال است. batch/none قبلی حفظ شده‌اند؛ حالت جدید ساختار پارامتر/buffer یکسان با batch ندارد. PointNet++ همچنان فقط batch/none دارد.
- InstanceNorm ماسک‌دار مشترک از track_running_stats=False استفاده می‌کند؛ حالت بدون mask native است. در خود utility حالت تماماً نامعتبر صفر می‌شود، ولی ورودی شبکه هندسهٔ بدون نقطهٔ معتبر را رد می‌کند.
- FNO instance سه‌بعدی و U-Net instance/group، و LayerNormهای Transolver/MeshGraphNet دست‌نخورده‌اند. InstanceNorm قبلی GCN روی ماتریس (N,C) نیز بازتعریف نشده است؛ آن را normalization مستقلِ نقاط هر گراف معرفی نمی‌کنیم.

## ۸. packing گراف و تفاوت معنی batch

nodeها و ویژگی یال‌ها به هم متصل و index هر گراف با تعداد رأس‌های قبلی offset می‌شود. index محلی **قبل از offset** بررسی می‌شود؛ معتبر بودن index در گراف packed برای معتبر بودن در گراف خودش کافی نیست. الگوریتم یک بار روی اجتماع گراف‌های بدون یال مشترک اجرا و نتیجه با طول‌ها جدا می‌شود؛ padding نداریم.

نبود یال بین دو گراف، استقلال آماری BN را تضمین نمی‌کند. GCN/GAT در training با BN از تمام رأس‌های packed آمار مشترک می‌گیرند؛ گراف ۴۰ رأسی دو برابر گراف ۲۰ رأسی سهم دارد. این با اجرای دو forward مستقل فرق می‌کند. در eval با آمار ثابت coupling آماری وجود ندارد.

MeshGraphNet فعلی BN ندارد؛ فقط layer_norm/none دارد و normalization هر رأس/یال روی ویژگی‌های خودش است. batch vector قدیمی ModelInput جزو binding موتور نیست؛ lengths/offsetها داخلی‌اند. معماری‌ای با graph-wise pooling مبتنی بر batch به توسعهٔ helper نیاز دارد.

## ۹. تغییرهای اختصاصی هر معماری

تمام شبکه‌ها تغییر مشترک import، initِ Model، signature، نگاشت Quantity و انتقال validation/محاسبات به evaluate را دارند. موارد زیر افزون بر آن‌اند:

| معماری | تغییر اختصاصی / بخش حفظ‌شده |
|---|---|
| PointNet | factory normalization، instance، mask scope و پاک‌کردن ورودی نامعتبر اضافه شده‌اند. بدنهٔ forward encoder، T-Net، headها و regularizer اصلی حفظ شده است. |
| PointNet++ | FPS به‌جای بافر inplace از list+stack استفاده می‌کند تا vmap کار کند؛ بررسی ماسک تهیِ helper مستقل اکنون با check برقرار است. interpolation و grouping اصلی حفظ و mask normalization سطح‌بندی شده است. |
| GINO | source و query به‌جای boolean slicing طول ثابت دارند؛ مختصات/feature نامعتبر پاک و سهم source در وزن kernel صفر می‌شود. default query نامعتبر بعد از محاسبه صفر می‌شود. فرمول kernel، projection، cutoff و weighted integral حفظ شده‌اند؛ target خالی نیز guard دارد. |
| DeepNURBS | endpoint_basis و where به‌جای شاخهٔ torch.any برای endpoint راست؛ تعریف NURBS، admissible/lifting و معنای latentِ پیش از قید حفظ شده‌اند. geometry هنوز در config/بافرهای شبکه است. |
| FNO3D | فقط BN بلوک به کلاس مشترک وصل شده؛ spectral.py، modeها، lifting، coordinate grid، skip و projection حفظ شده‌اند. |
| UNet3D | factory گزینهٔ batch از BN مشترک استفاده می‌کند؛ encoder/decoder، skipها و upsampling اصلی حفظ شده‌اند. محورهای latent با bottleneck نام‌گذاری شده‌اند. |
| GCN | convolution، درجه‌ها، self-loop و محاسبات اصلی حفظ شده‌اند؛ BN مشترک و packing افزوده شده‌اند. latent جدا در منبع نبود و اضافه نشده است. |
| GAT | incoming softmax، چند head، self-loop و محاسبهٔ attention حفظ شده‌اند؛ BN مشترک و packing بیرونی‌اند. |
| MeshGraphNet | encoder، processor، decoder و blocks.py بدون تغییر متن Python هستند؛ edge-first، receiver-sum و residualهای اصلی حفظ شده‌اند. |
| SIREN | SineLayer، frequencyها و initialization اصلی حفظ شده‌اند؛ محاسبهٔ نقطه‌ای evaluate با signature مستقل اجرا می‌شود. |
| Transolver | block.py، components.py و physics_attention.py بدون تغییر متن هستند؛ embedding، placeholder، slicing attention و head اصلی حفظ شده‌اند. |

### checkpoint و هزینهٔ ماسک در GINO

در منبع، query chunkهای integral داخل kernel checkpoint می‌شدند. اکنون GINO دارای checkpoint_batches=True است و Model کل فراخوانی batch‌شده را **بیرون تمام vmapها** با use_reentrant=False checkpoint می‌کند؛ kernel در اجرای شبکه use_checkpoint=False می‌گیرد. مسیر predict و checkpoint داخلیِ جداگانهٔ آن حذف شده‌اند.

در لیست برای هر عضو checkpoint جدا و در نبود batch نیز روی core اعمال می‌شود؛ no_grad آن را کنار می‌گذارد. دیگر شبکه‌ها این گزینه را پیش‌فرض فعال ندارند. محدودیت workspace زوج‌نقاط/chunkها حفظ شده است.

این تصمیم saved tensorها را کاهش می‌دهد ولی replay می‌تواند projection، Fourier blocks و decoder را هم تکرار کند؛ سرعت یا peak حافظهٔ مشخص تضمین نمی‌شود. slot ماسک‌شده نیز ممکن است kernel/decoder را طی کند؛ source_count کامل، chunk size و تعداد عملیات می‌تواند با slicing منبع فرق کند. اعتبارسنجی check نیز هنگام replay ممکن است دوباره اجرا شود. فعال‌سازی بی‌قید همین سیاست برای BN دارای state مناسب نیست.

## ۱۰. سازگاری، محدودیت‌ها و آزمون‌ها

- API سازنده و فراخوانی با نسخهٔ اصلی ناسازگار است؛ importهای `models` باید به `PINNCore.networks` و داده‌ها به Quantity/Value تبدیل شوند. حذف predict عمدی است.
- state_dict برای configهای قبلی نام پارامترهای اصلی را نگه می‌دارد؛ Solver پیشوند ثبت خودش مانند part_0. را اضافه می‌کند. گزینهٔ جدید instance و pickle کامل استثناهای توضیح‌داده‌شده‌اند.
- انتخاب خروجی، helperهای Python، آماده‌سازی batch و gather/cat/scatter normalization هنوز هزینه دارند. حذف تمام محاسبات غیرلازم، موازی‌سازی CPU/GPU، دامنهٔ هندسی و مشتق اختصاصی معماری‌ها پیاده نشده‌اند.
- آخرین تست کامل کد e256e93: **۲۸۳ تست موفق روی CPU**، شامل قرارداد مستقیم Model، vmap تودرتو، mask/NaN، آمار BN، list، packing، اتصال واقعی GeoCore، آموزش و مشتق‌های تا مرتبهٔ سوم GINO. یک FutureWarning موجود مربوط به TorchScript/Python 3.14 باقی است.
- در همین بازبینی، **نسخهٔ فعلی هر ۱۱ شبکه مستقیماً با سورس اصلی snapshot مقایسه شد**: config نمونهٔ کوچک در eval، وزن یکسان، خروجی prediction/latent، بارگذاری strict state_dict و گرادیان پارامترها همگی برابر بودند. اسکریپت یک‌بارهٔ .build-tmp/audit_current_vs_upstream.py ابزار بررسی محلی است و جزو تست نصب‌شده نیست.
- تست دائمی numeric_result در tests/test_networks.py، API فعلی را با چیدمان‌های متفاوت خودش مقایسه می‌کند؛ oracle مستقل upstream نیست. مقایسهٔ مستقل بند قبل شاهد جداگانه است.
- این شواهد اثبات برابری همهٔ configها، آمار ماسک‌دار قبلی، توالی RNG، GPU یا fullgraph compilation نیستند؛ تغییر معنای mask و گزینهٔ instance عمداً رفتار تازه‌اند.

## ۱۱. موضوع‌های مناسب برای مرور با همکار

۱. پذیرش وابستگی مستقیم شبکه‌ها به Model و حذف کامل API عددی قبلی، با وجود حفظ کد الگوریتم‌ها.
۲. قرار داشتن سیاست نام/محور معماری‌ها در helper مرکزی و شیوهٔ افزودن معماری تازه.
۳. آمار مشترک BN بین گراف‌ها و وزن بیشتر هندسهٔ دارای نمونهٔ بیشتر.
۴. mask-aware normalization، وزن همسایهٔ تکراری PointNet++ و سیاست instance بعد از pooling.
۵. benchmark اندازهٔ واقعی برای ماسک ثابت و checkpoint بزرگ‌تر GINO.
۶. آیندهٔ خروجی‌های auxiliary/metadata، domain و ابزارهای adaptation/rollout مرجع.

این‌ها لزوماً خطا نیستند؛ تصمیم‌های قابل مشاهدهٔ نسخهٔ ما هستند و نباید به‌عنوان انتخاب خود همکار معرفی شوند.

## ۱۲. فهرست کامل فایل‌ها و diff منبع

مسیر جدول نسبت به models اصلی و networks فعلی است. شمارش شامل ۴۳ فایل Python اصلی است؛ دو helper شبکه و check جدا اضافه شده‌اند. diffها تمام hunkهای تغییر را نشان می‌دهند؛ پایان خط و فاصلهٔ انتهای خط برای نمایش Markdown نرمال شده‌اند.

| فایل | وضعیت |
|---|---|
| `__init__.py` | تغییرکرده |
| `base.py` | حذف‌شده |
| `configs.py` | یکسان |
| `deep_nurbs/__init__.py` | تغییرکرده |
| `deep_nurbs/model.py` | تغییرکرده |
| `fno/__init__.py` | تغییرکرده |
| `fno/components.py` | تغییرکرده |
| `fno/model.py` | تغییرکرده |
| `fno/spectral.py` | یکسان |
| `gino/__init__.py` | تغییرکرده |
| `gino/components.py` | تغییرکرده |
| `gino/model.py` | تغییرکرده |
| `graph/__init__.py` | تغییرکرده |
| `graph/base.py` | حذف‌شده |
| `graph/gat.py` | تغییرکرده |
| `graph/gcn.py` | تغییرکرده |
| `graph/layers.py` | تغییرکرده |
| `graph/meshgraphnet/__init__.py` | یکسان |
| `graph/meshgraphnet/blocks.py` | یکسان |
| `graph/meshgraphnet/decoder.py` | یکسان |
| `graph/meshgraphnet/encoder.py` | یکسان |
| `graph/meshgraphnet/model.py` | تغییرکرده |
| `graph/meshgraphnet/processor.py` | یکسان |
| `pointnet/__init__.py` | تغییرکرده |
| `pointnet/components.py` | تغییرکرده |
| `pointnet/model.py` | تغییرکرده |
| `pointnet2/__init__.py` | تغییرکرده |
| `pointnet2/components.py` | تغییرکرده |
| `pointnet2/model.py` | تغییرکرده |
| `registry.py` | تغییرکرده |
| `representations.py` | یکسان |
| `siren/__init__.py` | تغییرکرده |
| `siren/layers.py` | یکسان |
| `siren/model.py` | تغییرکرده |
| `transolver/__init__.py` | تغییرکرده |
| `transolver/block.py` | یکسان |
| `transolver/components.py` | یکسان |
| `transolver/model.py` | تغییرکرده |
| `transolver/physics_attention.py` | یکسان |
| `types.py` | یکسان |
| `unet3d/__init__.py` | تغییرکرده |
| `unet3d/components.py` | تغییرکرده |
| `unet3d/model.py` | تغییرکرده |

<details>
<summary>__init__.py</summary>

```diff
--- third_party/neural3dbench/models/__init__.py
+++ src/PINNCore/networks/__init__.py
@@ -1,4 +1,3 @@
-from .base import BaseNetwork
 from .configs import BaseModelConfig
 from .deep_nurbs import DeepNURBS, DeepNURBSConfig
 from .fno import FNO3D, FNO3DConfig, SpectralConv3d
@@ -22,7 +21,6 @@

 __all__ = [
     "BaseModelConfig",
-    "BaseNetwork",
     "CoordinateRepresentationAdapter",
     "DeepNURBS",
     "DeepNURBSConfig",
```

</details>

<details>
<summary>base.py</summary>

```diff
--- third_party/neural3dbench/models/base.py
+++ /dev/null
@@ -1,75 +0,0 @@
-from __future__ import annotations
-
-from abc import ABC, abstractmethod
-from typing import Any
-
-import torch
-from torch import nn
-
-from .configs import BaseModelConfig
-from .types import ModelCapabilities, ModelInput, ModelOutput
-
-
-class BaseNetwork(nn.Module, ABC):
-    """Stable interface every model must satisfy."""
-
-    config_class = BaseModelConfig
-
-    def __init__(self, config: BaseModelConfig | None = None):
-        super().__init__()
-        self.config = config or self.config_class()
-
-    @property
-    @abstractmethod
-    def required_inputs(self) -> set[str]:
-        raise NotImplementedError
-
-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation=self.config.representation,
-            requires=set(self.required_inputs),
-            supports=set(),
-            variable_geometry=False,
-            variable_node_count=False,
-        )
-
-    @property
-    def supported_representations(self) -> set[str]:
-        rep = self.capabilities.representation
-        return {rep} if rep is not None else set()
-
-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        if not isinstance(inputs, ModelInput):
-            raise TypeError(f"Expected ModelInput, got {type(inputs).__name__}.")
-
-        required = set(self.required_inputs)
-        available = inputs.available_fields()
-        missing = required - available
-        if missing:
-            missing_fields = ", ".join(sorted(missing))
-            raise ValueError(
-                f"{self.__class__.__name__} requires inputs: {missing_fields}. "
-                f"Available: {sorted(available)}."
-            )
-
-        rep = self.capabilities.representation
-        if rep is not None and inputs.representation is not None and inputs.representation != rep:
-            raise ValueError(
-                f"Model expects representation '{rep}' but received '{inputs.representation}'."
-            )
-
-        return True
-
-    @abstractmethod
-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        raise NotImplementedError
-
-    def predict(self, inputs: ModelInput) -> ModelOutput:
-        return self(inputs)
-
-    @staticmethod
-    def _ensure_tensor(value: Any, *, name: str = "tensor") -> torch.Tensor:
-        if isinstance(value, torch.Tensor):
-            return value
-        return torch.as_tensor(value, dtype=torch.float32)
```

</details>

<details>
<summary>deep_nurbs/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/deep_nurbs/__init__.py
+++ src/PINNCore/networks/deep_nurbs/__init__.py
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
+++ src/PINNCore/networks/deep_nurbs/model.py
@@ -4,12 +4,14 @@
 from numbers import Integral
 from typing import Any, Mapping, Sequence

+from PINNCore.quantities import Quantity
+
 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore import Model, check
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig


 def _positive_int(value: int, name: str) -> int:
@@ -224,20 +226,33 @@

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


-class DeepNURBS(BaseNetwork):
+class DeepNURBS(Model):
     """NURBS-admissible neural approximation on a 2D parametric surface."""

     config_class = DeepNURBSConfig
-
-    def __init__(self, config: DeepNURBSConfig | None = None) -> None:
-        super().__init__(config=config or DeepNURBSConfig())
+    vmap_randomness = "different"
+
+    @property
+    def signature(self):
+        return contract_signature(self)
+
+    def __init__(
+        self,
+        config: DeepNURBSConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__()
+        self.config = config or DeepNURBSConfig()
+        initialize_contract(self, inputs, outputs)
         control_points = torch.tensor(self.config.control_points, dtype=torch.float32)
         weights = torch.tensor(self.config.weights, dtype=torch.float32)
         admissible = torch.tensor(
@@ -260,37 +275,6 @@
             input_dim = self.config.hidden_dim
         self.output_layer = nn.Linear(input_dim, self.config.output_dim)

-    @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates"}
-
-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="PARAMETRIC_SURFACE",
-            requires=set(self.required_inputs),
-            supports=set(),
-            variable_geometry=False,
-            variable_node_count=True,
-        )
-
-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
-            raise ValueError(
-                "DeepNURBS coordinates must have shape [B, N, 2], "
-                f"got {tuple(coordinates.shape)}."
-            )
-        if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
-            raise ValueError("coordinates must contain at least one batch and one point.")
-        if not coordinates.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        if not torch.isfinite(coordinates).all():
-            raise ValueError("coordinates must be finite.")
-        if torch.any((coordinates < 0.0) | (coordinates > 1.0)):
-            raise ValueError("parametric coordinates must lie in [0, 1].")
-        return True

     def _evaluate_nurbs(
         self, parametric_coordinates: torch.Tensor
@@ -323,31 +307,52 @@
         )
         return physical_coordinates, admissible_field, lifting_field

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        parametric_coordinates = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        physical_coordinates, admissible_field, lifting_field = self._evaluate_nurbs(
-            parametric_coordinates
-        )
-        neural_field = physical_coordinates
-        for layer in self.hidden_layers:
-            neural_field = layer(neural_field)
-            if self.config.activation == "cubic_relu":
-                neural_field = torch.relu(neural_field).pow(3)
-            else:
-                neural_field = torch.tanh(neural_field)
-        neural_field = self.output_layer(neural_field)
-        predictions = admissible_field * neural_field + lifting_field
-        return ModelOutput(
-            predictions=predictions,
-            latent=neural_field,
-            metadata={
-                "parametric_coordinates": parametric_coordinates,
-                "physical_coordinates": physical_coordinates,
-                "admissible_field": admissible_field,
-                "lifting_field": lifting_field,
-            },
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            coordinates = inputs["coordinates"]
+            if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
+                raise ValueError(
+                    f"DeepNURBS coordinates must have shape [B, N, 2], got {tuple(coordinates.shape)}."
+                )
+            if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
+                raise ValueError(
+                    "coordinates must contain at least one batch and one point."
+                )
+            if not coordinates.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            check(torch.isfinite(coordinates).all(), "coordinates must be finite.")
+            check(
+                ~torch.any((coordinates < 0.0) | (coordinates > 1.0)),
+                "parametric coordinates must lie in [0, 1].",
+            )
+            parameter = next(self.parameters())
+            parametric_coordinates = torch.as_tensor(
+                inputs["coordinates"], device=parameter.device, dtype=parameter.dtype
+            )
+            physical_coordinates, admissible_field, lifting_field = self._evaluate_nurbs(
+                parametric_coordinates
+            )
+            neural_field = physical_coordinates
+            for layer in self.hidden_layers:
+                neural_field = layer(neural_field)
+                if self.config.activation == "cubic_relu":
+                    neural_field = torch.relu(neural_field).pow(3)
+                else:
+                    neural_field = torch.tanh(neural_field)
+            neural_field = self.output_layer(neural_field)
+            predictions = admissible_field * neural_field + lifting_field
+            results.append(
+                {
+                    "predictions": predictions,
+                    "latent": neural_field,
+                    "metadata": {
+                        "parametric_coordinates": parametric_coordinates,
+                        "physical_coordinates": physical_coordinates,
+                        "admissible_field": admissible_field,
+                        "lifting_field": lifting_field,
+                    },
+                }
+            )
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>fno/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/fno/__init__.py
+++ src/PINNCore/networks/fno/__init__.py
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
+++ src/PINNCore/networks/fno/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from PINNCore.networks.normalization import BatchNorm3d

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
+++ src/PINNCore/networks/fno/model.py
@@ -1,15 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
-from typing import Sequence
+from typing import Mapping, Sequence

 import torch
 from torch import nn
 from torch.nn import functional as F

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.quantities import Quantity
+
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig

 from .components import (
     FourierOperatorBlock3d,
@@ -72,13 +74,22 @@
             raise ValueError("dropout must be in [0, 1).")


-class FNO3D(BaseNetwork):
+class FNO3D(Model):
     """Fourier Neural Operator for dense fields on regular 3D grids."""

     config_class = FNO3DConfig
+    vmap_randomness = "different"

-    def __init__(self, config: FNO3DConfig | None = None) -> None:
-        super().__init__(config=config or FNO3DConfig())
+    def __init__(
+        self,
+        config: FNO3DConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__()
+        self.config = config or FNO3DConfig()
+        initialize_contract(self, inputs, outputs)
         lifted_channels = self.config.input_channels + (
             3 if self.config.append_coordinates else 0
         )
@@ -101,68 +112,58 @@
         )

     @property
-    def required_inputs(self) -> set[str]:
-        return {"voxel_fields"}
+    def signature(self):
+        return contract_signature(self)

-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="GRID",
-            requires={"voxel_fields"},
-            supports={"boundary_mask", "material_properties"},
-            variable_geometry=False,
-            variable_node_count=False,
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            volume = inputs["voxel_fields"]
+            if volume.ndim != 5:
+                raise ValueError(
+                    "FNO3D voxel_fields must have shape [B, C, D, H, W], "
+                    f"got {tuple(volume.shape)}."
+                )
+            if volume.shape[1] != self.config.input_channels:
+                raise ValueError(
+                    f"FNO3D expects {self.config.input_channels} input channels, "
+                    f"got {volume.shape[1]}."
+                )
+            if not volume.is_floating_point():
+                raise TypeError("voxel_fields must use a floating-point dtype.")
+            # Validate against the user-supplied grid, not an artificially enlarged
+            # padded grid, so incompatible data fails early and predictably.
+            self.blocks[0].spectral._validate_spatial_shape(volume.shape[-3:])

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        volume = self._ensure_tensor(inputs.voxel_fields, name="voxel_fields")
-        if volume.ndim != 5:
-            raise ValueError(
-                "FNO3D voxel_fields must have shape [B, C, D, H, W], "
-                f"got {tuple(volume.shape)}."
+            parameter = next(self.parameters())
+            x = torch.as_tensor(
+                inputs.get("voxel_fields"), device=parameter.device, dtype=parameter.dtype
             )
-        if volume.shape[1] != self.config.input_channels:
-            raise ValueError(
-                f"FNO3D expects {self.config.input_channels} input channels, "
-                f"got {volume.shape[1]}."
-            )
-        if not volume.is_floating_point():
-            raise TypeError("voxel_fields must use a floating-point dtype.")
-        # Validate against the user-supplied grid, not an artificially enlarged
-        # padded grid, so incompatible data fails early and predictably.
-        self.blocks[0].spectral._validate_spatial_shape(volume.shape[-3:])
-        return True
+            original_shape = tuple(x.shape[-3:])
+            if self.config.append_coordinates:
+                x = torch.cat((x, normalized_coordinate_grid(x)), dim=1)
+            x = self.lifting(x)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        x = torch.as_tensor(
-            inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
-        )
-        original_shape = tuple(x.shape[-3:])
-        if self.config.append_coordinates:
-            x = torch.cat((x, normalized_coordinate_grid(x)), dim=1)
-        x = self.lifting(x)
+            pad_d, pad_h, pad_w = self.config.padding
+            if any(self.config.padding):
+                # F.pad order is W, H, D. Padding is one-sided at the upper end.
+                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
+            for index, block in enumerate(self.blocks):
+                x = block(x)
+                if index < len(self.blocks) - 1:
+                    x = self.activation(x)
+            if any(self.config.padding):
+                depth, height, width = original_shape
+                x = x[..., :depth, :height, :width]

-        pad_d, pad_h, pad_w = self.config.padding
-        if any(self.config.padding):
-            # F.pad order is W, H, D. Padding is one-sided at the upper end.
-            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
-        for index, block in enumerate(self.blocks):
-            x = block(x)
-            if index < len(self.blocks) - 1:
-                x = self.activation(x)
-        if any(self.config.padding):
-            depth, height, width = original_shape
-            x = x[..., :depth, :height, :width]
-
-        predictions = self.projection(x)
-        return ModelOutput(
-            predictions=predictions,
-            latent=x,
-            metadata={
-                "spatial_shape": tuple(predictions.shape[-3:]),
-                "modes": self.config.modes,
-            },
-        )
+            predictions = self.projection(x)
+            results.append(dict(
+                predictions=predictions,
+                latent=x,
+                metadata={
+                    "spatial_shape": tuple(predictions.shape[-3:]),
+                    "modes": self.config.modes,
+                },
+            ))
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>gino/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/gino/__init__.py
+++ src/PINNCore/networks/gino/__init__.py
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
+++ src/PINNCore/networks/gino/components.py
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
+++ src/PINNCore/networks/gino/model.py
@@ -1,16 +1,19 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
 from numbers import Integral
 from typing import Sequence

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.fno.components import FourierOperatorBlock3d, get_activation
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore import Model, check
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig
+from ..fno.components import FourierOperatorBlock3d, get_activation

 from .components import KernelIntegralTransform

@@ -89,13 +92,30 @@
             raise ValueError(f"modes[2]={modes_w} exceeds latent W//2+1={width // 2 + 1}.")


-class GINO(BaseNetwork):
+class GINO(Model):
     """Geometry-aware point-to-grid, Fourier, and grid-to-point operator."""

     config_class = GINOConfig
-
-    def __init__(self, config: GINOConfig | None = None) -> None:
-        super().__init__(config=config or GINOConfig())
+    vmap_randomness = "different"
+
+    @property
+    def signature(self):
+        return contract_signature(self)
+
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
+        super().__init__()
+        self.config = config or GINOConfig()
+        initialize_contract(self, inputs, outputs)
         self.encoder = KernelIntegralTransform(
             self.config.input_dim,
             self.config.width,
@@ -119,82 +139,6 @@
         )
         self.output_projection = nn.Linear(self.config.width, self.config.output_dim)

-    @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates", "point_features"}
-
-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="POINT_CLOUD",
-            requires=set(self.required_inputs),
-            supports={"point_mask", "query_coordinates"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
-
-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        features = self._ensure_tensor(inputs.point_features, name="point_features")
-        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
-            raise ValueError(
-                "GINO coordinates must have shape [B, N, 3], "
-                f"got {tuple(coordinates.shape)}."
-            )
-        if coordinates.shape[1] == 0:
-            raise ValueError("Each source point cloud must contain at least one point.")
-        if not coordinates.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        expected_features = (*coordinates.shape[:2], self.config.input_dim)
-        if tuple(features.shape) != expected_features:
-            raise ValueError(
-                f"point_features must have shape {expected_features}, got {tuple(features.shape)}."
-            )
-        if not features.is_floating_point():
-            raise TypeError("point_features must use a floating-point dtype.")
-
-        if inputs.point_mask is None:
-            source_mask = torch.ones(coordinates.shape[:2], dtype=torch.bool, device=coordinates.device)
-        else:
-            source_mask = torch.as_tensor(inputs.point_mask, device=coordinates.device)
-            if tuple(source_mask.shape) != tuple(coordinates.shape[:2]):
-                raise ValueError(
-                    f"point_mask must have shape {tuple(coordinates.shape[:2])}, "
-                    f"got {tuple(source_mask.shape)}."
-                )
-            if source_mask.dtype != torch.bool:
-                raise TypeError("point_mask must use a boolean dtype.")
-        if not torch.all(source_mask.any(dim=1)):
-            raise ValueError("Every point cloud must contain at least one valid source point.")
-        if not torch.isfinite(coordinates[source_mask]).all():
-            raise ValueError("Valid source coordinates must be finite.")
-        if not torch.isfinite(features[source_mask]).all():
-            raise ValueError("Valid point_features must be finite.")
-        valid_source_coordinates = coordinates[source_mask]
-        if torch.any((valid_source_coordinates < 0.0) | (valid_source_coordinates > 1.0)):
-            raise ValueError("Valid source coordinates must be normalized to [0, 1].")
-
-        if inputs.query_coordinates is not None:
-            query_coordinates = self._ensure_tensor(
-                inputs.query_coordinates, name="query_coordinates"
-            )
-            if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 3:
-                raise ValueError(
-                    "query_coordinates must have shape [B, Q, 3], "
-                    f"got {tuple(query_coordinates.shape)}."
-                )
-            if query_coordinates.shape[0] != coordinates.shape[0]:
-                raise ValueError("query_coordinates batch size must match coordinates.")
-            if query_coordinates.shape[1] == 0:
-                raise ValueError("Each query set must contain at least one point.")
-            if not query_coordinates.is_floating_point():
-                raise TypeError("query_coordinates must use a floating-point dtype.")
-            if not torch.isfinite(query_coordinates).all():
-                raise ValueError("query_coordinates must be finite.")
-            if torch.any((query_coordinates < 0.0) | (query_coordinates > 1.0)):
-                raise ValueError("query_coordinates must be normalized to [0, 1].")
-        return True

     def _latent_coordinates(self, reference: torch.Tensor) -> torch.Tensor:
         axes = [
@@ -204,73 +148,153 @@
         grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
         return grid.reshape(-1, 3)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        coordinates = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        features = torch.as_tensor(
-            inputs.point_features, device=parameter.device, dtype=parameter.dtype
-        )
-        source_mask = (
-            torch.ones(coordinates.shape[:2], dtype=torch.bool, device=parameter.device)
-            if inputs.point_mask is None
-            else torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
-        )
-        default_queries = inputs.query_coordinates is None
-        queries = coordinates if default_queries else torch.as_tensor(
-            inputs.query_coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        latent_coordinates = self._latent_coordinates(coordinates)
-        latent_batches = []
-
-        for batch_index in range(coordinates.shape[0]):
-            valid = source_mask[batch_index]
-            latent = self.encoder(
-                coordinates[batch_index, valid],
-                latent_coordinates,
-                features[batch_index, valid],
-            )
-            latent = latent.reshape(*self.config.latent_resolution, self.config.width)
-            latent_batches.append(latent.permute(3, 0, 1, 2))
-
-        latent_grid = torch.stack(latent_batches, dim=0)
-        for index, block in enumerate(self.blocks):
-            latent_grid = block(latent_grid)
-            if index < len(self.blocks) - 1:
-                latent_grid = self.activation(latent_grid)
-
-        flat_latent = latent_grid.permute(0, 2, 3, 4, 1).reshape(
-            coordinates.shape[0], -1, self.config.width
-        )
-        decoded_batches = []
-        for batch_index in range(coordinates.shape[0]):
-            query_mask = source_mask[batch_index] if default_queries else None
-            valid_queries = queries[batch_index] if query_mask is None else queries[batch_index, query_mask]
-            decoded = self.decoder(
-                latent_coordinates,
-                valid_queries,
-                flat_latent[batch_index],
-            )
-            projected = self.output_projection(decoded)
-            if query_mask is not None:
-                batch_predictions = projected.new_zeros(
-                    (queries.shape[1], self.config.output_dim)
-                )
-                batch_predictions[query_mask] = projected
-                decoded_batches.append(batch_predictions)
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            coordinates = inputs["coordinates"]
+            features = inputs["point_features"]
+            if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
+                raise ValueError(
+                    f"GINO coordinates must have shape [B, N, 3], got {tuple(coordinates.shape)}."
+                )
+            if coordinates.shape[1] == 0:
+                raise ValueError("Each source point cloud must contain at least one point.")
+            if not coordinates.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            expected_features = (*coordinates.shape[:2], self.config.input_dim)
+            if tuple(features.shape) != expected_features:
+                raise ValueError(
+                    f"point_features must have shape {expected_features}, got {tuple(features.shape)}."
+                )
+            if not features.is_floating_point():
+                raise TypeError("point_features must use a floating-point dtype.")
+            if inputs.get("point_mask") is None:
+                source_mask = torch.ones(
+                    coordinates.shape[:2], dtype=torch.bool, device=coordinates.device
+                )
             else:
-                decoded_batches.append(projected)
-
-        predictions = torch.stack(decoded_batches, dim=0)
-        return ModelOutput(
-            predictions=predictions,
-            latent=latent_grid,
-            metadata={
-                "query_coordinates": queries,
-                "query_is_source": default_queries,
-                "point_mask": source_mask,
-                "latent_resolution": self.config.latent_resolution,
-            },
-        )
+                source_mask = torch.as_tensor(
+                    inputs.get("point_mask"), device=coordinates.device
+                )
+                if tuple(source_mask.shape) != tuple(coordinates.shape[:2]):
+                    raise ValueError(
+                        f"point_mask must have shape {tuple(coordinates.shape[:2])}, got {tuple(source_mask.shape)}."
+                    )
+                if source_mask.dtype != torch.bool:
+                    raise TypeError("point_mask must use a boolean dtype.")
+            check(
+                source_mask.any(dim=1),
+                "Every point cloud must contain at least one valid source point.",
+            )
+            valid_coordinates = torch.where(source_mask.unsqueeze(-1), coordinates, 0.0)
+            valid_features = torch.where(source_mask.unsqueeze(-1), features, 0.0)
+            check(
+                torch.isfinite(valid_coordinates),
+                "Valid source coordinates must be finite.",
+            )
+            check(torch.isfinite(valid_features), "Valid point_features must be finite.")
+            check(
+                (valid_coordinates >= 0) & (valid_coordinates <= 1),
+                "Valid source coordinates must be normalized to [0, 1].",
+            )
+            if inputs.get("query_coordinates") is not None:
+                query_coordinates = inputs.get("query_coordinates")
+                if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 3:
+                    raise ValueError(
+                        f"query_coordinates must have shape [B, Q, 3], got {tuple(query_coordinates.shape)}."
+                    )
+                if query_coordinates.shape[0] != coordinates.shape[0]:
+                    raise ValueError("query_coordinates batch size must match coordinates.")
+                if query_coordinates.shape[1] == 0:
+                    raise ValueError("Each query set must contain at least one point.")
+                if not query_coordinates.is_floating_point():
+                    raise TypeError("query_coordinates must use a floating-point dtype.")
+                check(
+                    torch.isfinite(query_coordinates).all(),
+                    "query_coordinates must be finite.",
+                )
+                check(
+                    ~torch.any((query_coordinates < 0.0) | (query_coordinates > 1.0)),
+                    "query_coordinates must be normalized to [0, 1].",
+                )
+            parameter = next(self.parameters())
+            coordinates = torch.as_tensor(
+                inputs["coordinates"], device=parameter.device, dtype=parameter.dtype
+            )
+            features = torch.as_tensor(
+                inputs["point_features"], device=parameter.device, dtype=parameter.dtype
+            )
+            source_mask = (
+                torch.ones(coordinates.shape[:2], dtype=torch.bool, device=parameter.device)
+                if inputs.get("point_mask") is None
+                else torch.as_tensor(
+                    inputs.get("point_mask"), device=parameter.device, dtype=torch.bool
+                )
+            )
+            default_queries = inputs.get("query_coordinates") is None
+            queries = (
+                coordinates
+                if default_queries
+                else torch.as_tensor(
+                    inputs.get("query_coordinates"),
+                    device=parameter.device,
+                    dtype=parameter.dtype,
+                )
+            )
+            latent_coordinates = self._latent_coordinates(coordinates)
+            latent_batches = []
+            for batch_index in range(coordinates.shape[0]):
+                valid = source_mask[batch_index]
+                latent = self.encoder(
+                    coordinates[batch_index],
+                    latent_coordinates,
+                    features[batch_index],
+                    source_mask=valid,
+                    use_checkpoint=False,
+                )
+                latent = latent.reshape(*self.config.latent_resolution, self.config.width)
+                latent_batches.append(latent.permute(3, 0, 1, 2))
+            latent_grid = torch.stack(latent_batches, dim=0)
+            for index, block in enumerate(self.blocks):
+                latent_grid = block(latent_grid)
+                if index < len(self.blocks) - 1:
+                    latent_grid = self.activation(latent_grid)
+            flat_latent = latent_grid.permute(0, 2, 3, 4, 1).reshape(
+                coordinates.shape[0], -1, self.config.width
+            )
+            decoded_batches = []
+            for batch_index in range(coordinates.shape[0]):
+                query_mask = source_mask[batch_index] if default_queries else None
+                valid_queries = queries[batch_index]
+                if query_mask is not None:
+                    valid_queries = torch.where(
+                        query_mask.unsqueeze(-1), valid_queries, 0.0
+                    )
+                decoded = self.decoder(
+                    latent_coordinates,
+                    valid_queries,
+                    flat_latent[batch_index],
+                    use_checkpoint=False,
+                )
+                projected = self.output_projection(decoded)
+                if query_mask is not None:
+                    decoded_batches.append(
+                        projected.masked_fill(~query_mask.unsqueeze(-1), 0.0)
+                    )
+                else:
+                    decoded_batches.append(projected)
+            predictions = torch.stack(decoded_batches, dim=0)
+            results.append(
+                {
+                    "predictions": predictions,
+                    "latent": latent_grid,
+                    "metadata": {
+                        "query_coordinates": queries,
+                        "query_is_source": default_queries,
+                        "point_mask": source_mask,
+                        "latent_resolution": self.config.latent_resolution,
+                    },
+                }
+            )
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>graph/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/graph/__init__.py
+++ src/PINNCore/networks/graph/__init__.py
@@ -1,8 +1,7 @@
 from __future__ import annotations

-from models.registry import ModelRegistry
+from ..registry import ModelRegistry

-from .base import BaseGraphNetwork
 from .gat import GAT, GATConfig
 from .gcn import GCN, GCNConfig
 from .meshgraphnet import MeshGraphNet, MeshGraphNetConfig
@@ -12,7 +11,6 @@
 ModelRegistry.register("meshgraphnet", MeshGraphNet, MeshGraphNetConfig)

 __all__ = [
-    "BaseGraphNetwork",
     "GAT",
     "GATConfig",
     "GCN",
```

</details>

<details>
<summary>graph/base.py</summary>

```diff
--- third_party/neural3dbench/models/graph/base.py
+++ /dev/null
@@ -1,51 +0,0 @@
-from __future__ import annotations
-
-from abc import ABC
-
-import torch
-
-from models.base import BaseNetwork
-from models.types import ModelCapabilities
-
-
-class BaseGraphNetwork(BaseNetwork, ABC):
-    """Shared validation and capability metadata for graph-based models."""
-
-    @property
-    def required_inputs(self) -> set[str]:
-        return {"node_features", "edge_index"}
-
-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="GRAPH",
-            requires=set(self.required_inputs),
-            supports={"batch", "edge_features", "coordinates"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
-
-    def validate_inputs(self, inputs) -> bool:
-        super().validate_inputs(inputs)
-
-        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
-        if node_features.dim() != 2:
-            raise ValueError(f"node_features must have shape [N, F], got {tuple(node_features.shape)}.")
-        if node_features.size(0) == 0:
-            raise ValueError("node_features must contain at least one node.")
-
-        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
-        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
-            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
-        if edge_index.numel() > 0:
-            if edge_index.min() < 0 or edge_index.max() >= node_features.size(0):
-                raise ValueError(
-                    f"edge_index entries must be in the range [0, {node_features.size(0) - 1}] for N={node_features.size(0)}."
-                )
-
-        if inputs.batch is not None:
-            batch = self._ensure_tensor(inputs.batch, name="batch")
-            if batch.dim() != 1 or batch.numel() != node_features.size(0):
-                raise ValueError(f"batch must have shape [N], got {tuple(batch.shape)} for N={node_features.size(0)}.")
-
-        return True
```

</details>

<details>
<summary>graph/gat.py</summary>

```diff
--- third_party/neural3dbench/models/graph/gat.py
+++ src/PINNCore/networks/graph/gat.py
@@ -1,14 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
+from PINNCore.model import Model

 import torch
 from torch import nn

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ..configs import BaseModelConfig
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs

-from .base import BaseGraphNetwork
 from .layers import GATBlock


@@ -79,13 +82,26 @@
             )


-class GAT(BaseGraphNetwork):
+class GAT(Model):
     """Node-level Graph Attention Network with a fixed-width hidden state."""

     config_class = GATConfig
+    vmap_randomness = "different"

-    def __init__(self, config: GATConfig | None = None):
-        super().__init__(config=config or GATConfig())
+    @property
+    def signature(self):
+        return contract_signature(self)
+
+    def __init__(
+        self,
+        config: GATConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__()
+        self.config = config or GATConfig()
+        initialize_contract(self, inputs, outputs)
         self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
         self.layers = nn.ModuleList(
             GATBlock(
@@ -107,22 +123,6 @@
         )
         self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
-        if not node_features.is_floating_point():
-            raise TypeError("node_features must use a floating-point dtype.")
-        if node_features.shape[1] != self.config.input_dim:
-            raise ValueError(
-                f"GAT expects input feature dimension {self.config.input_dim}, "
-                f"got {node_features.shape[1]}."
-            )
-
-        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
-        if edge_index.dtype == torch.bool or edge_index.is_floating_point() or edge_index.is_complex():
-            raise TypeError("edge_index must use an integer dtype.")
-        return True
-
     def _prepare_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
         edge_index = torch.as_tensor(
             edge_index,
@@ -139,37 +139,48 @@
         self_loops = torch.stack((nodes, nodes), dim=0)
         return torch.cat((non_loop_edges, self_loops), dim=1)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(
-            device=self.input_proj.weight.device,
-            dtype=self.input_proj.weight.dtype,
-        )
-        edge_index = self._prepare_edge_index(inputs.edge_index, node_features.size(0))
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            node_features = inputs["node_features"]
+            if not node_features.is_floating_point():
+                raise TypeError("node_features must use a floating-point dtype.")
+            if node_features.shape[1] != self.config.input_dim:
+                raise ValueError(
+                    f"GAT expects input feature dimension {self.config.input_dim}, "
+                    f"got {node_features.shape[1]}."
+                )
+            node_features = node_features.to(
+                device=self.input_proj.weight.device,
+                dtype=self.input_proj.weight.dtype,
+            )
+            edge_index = self._prepare_edge_index(inputs["edge_index"], node_features.size(0))

-        hidden = self.input_proj(node_features)
-        attention_weights = [] if self.config.return_attention_weights else None
-        for layer in self.layers:
-            if attention_weights is None:
-                hidden = layer(hidden, edge_index)
-            else:
-                hidden, coefficients = layer(
-                    hidden,
-                    edge_index,
-                    return_attention_weights=True,
-                )
-                attention_weights.append(
-                    {"edge_index": edge_index, "coefficients": coefficients}
-                )
+            hidden = self.input_proj(node_features)
+            attention_weights = [] if self.config.return_attention_weights else None
+            for layer in self.layers:
+                if attention_weights is None:
+                    hidden = layer(hidden, edge_index)
+                else:
+                    hidden, coefficients = layer(
+                        hidden,
+                        edge_index,
+                        return_attention_weights=True,
+                    )
+                    attention_weights.append(
+                        {"edge_index": edge_index, "coefficients": coefficients}
+                    )

-        predictions = self.output_proj(hidden)
-        auxiliary = None
-        if attention_weights is not None:
-            auxiliary = {"attention_weights": attention_weights}
-        metadata = {"batch": inputs.batch} if inputs.batch is not None else {}
-        return ModelOutput(
-            predictions=predictions,
-            latent=hidden,
-            auxiliary=auxiliary,
-            metadata=metadata,
-        )
+            predictions = self.output_proj(hidden)
+            auxiliary = None
+            if attention_weights is not None:
+                auxiliary = {"attention_weights": attention_weights}
+            metadata = {"batch": inputs["batch"]} if inputs.get("batch") is not None else {}
+            results.append({
+                "predictions": predictions,
+                "latent": hidden,
+                "auxiliary": auxiliary,
+                "metadata": metadata,
+            })
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>graph/gcn.py</summary>

```diff
--- third_party/neural3dbench/models/graph/gcn.py
+++ src/PINNCore/networks/graph/gcn.py
@@ -1,14 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
+from PINNCore.model import Model

 import torch
 from torch import nn

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ..configs import BaseModelConfig
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs

-from .base import BaseGraphNetwork
 from .layers import GCNBlock


@@ -50,11 +53,24 @@
             raise ValueError(f"Unknown normalization '{self.normalization}'. Available: {sorted(valid_normalizations)}")


-class GCN(BaseGraphNetwork):
+class GCN(Model):
     config_class = GCNConfig
+    vmap_randomness = "different"

-    def __init__(self, config: GCNConfig | None = None):
-        super().__init__(config=config or GCNConfig())
+    @property
+    def signature(self):
+        return contract_signature(self)
+
+    def __init__(
+        self,
+        config: GCNConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__()
+        self.config = config or GCNConfig()
+        initialize_contract(self, inputs, outputs)

         self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)

@@ -75,25 +91,6 @@

         self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

-    @property
-    def required_inputs(self) -> set[str]:
-        return {"node_features", "edge_index"}
-
-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-
-        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
-        if node_features.shape[1] != self.config.input_dim:
-            raise ValueError(
-                f"GCN expects input feature dimension {self.config.input_dim}, got {node_features.shape[1]}."
-            )
-
-        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
-        if edge_index.shape[0] != 2:
-            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
-
-        return True
-
     def _prepare_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
         edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=self.input_proj.weight.device)
         if edge_index.dim() != 2 or edge_index.shape[0] != 2:
@@ -105,14 +102,21 @@

         return edge_index

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            node_features = inputs["node_features"]
+            if node_features.shape[1] != self.config.input_dim:
+                raise ValueError(
+                    f"GCN expects input feature dimension {self.config.input_dim}, got {node_features.shape[1]}."
+                )
+            node_features = node_features.to(self.input_proj.weight.device)
+            edge_index = self._prepare_edge_index(inputs["edge_index"], node_features.size(0))

-        node_features = self._ensure_tensor(inputs.node_features, name="node_features").to(self.input_proj.weight.device)
-        edge_index = self._prepare_edge_index(inputs.edge_index, node_features.size(0))
-
-        hidden = self.input_proj(node_features)
-        for layer in self.layers:
-            hidden = layer(hidden, edge_index)
-        predictions = self.output_proj(hidden)
-        return ModelOutput(predictions=predictions)
+            hidden = self.input_proj(node_features)
+            for layer in self.layers:
+                hidden = layer(hidden, edge_index)
+            predictions = self.output_proj(hidden)
+            results.append({"predictions": predictions})
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>graph/layers.py</summary>

```diff
--- third_party/neural3dbench/models/graph/layers.py
+++ src/PINNCore/networks/graph/layers.py
@@ -2,6 +2,7 @@

 import torch
 from torch import nn
+from PINNCore.networks.normalization import BatchNorm1d


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
+++ src/PINNCore/networks/graph/meshgraphnet/model.py
@@ -1,13 +1,16 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
+from PINNCore.model import Model

 import torch

-from models.configs import BaseModelConfig
-from models.types import ModelInput, ModelOutput
+from ...configs import BaseModelConfig
+from ..._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs

-from ..base import BaseGraphNetwork
 from .decoder import MeshGraphNetDecoder
 from .encoder import MeshGraphNetEncoder
 from .processor import MeshGraphNetProcessor
@@ -63,7 +66,7 @@
             )


-class MeshGraphNet(BaseGraphNetwork):
+class MeshGraphNet(Model):
     """Encode-process-decode graph network for per-node physical predictions.

     Architectural decisions follow Pfaff et al. (ICLR 2021), DeepMind's
@@ -75,9 +78,22 @@
     """

     config_class = MeshGraphNetConfig
+    vmap_randomness = "different"

-    def __init__(self, config: MeshGraphNetConfig | None = None) -> None:
-        super().__init__(config=config or MeshGraphNetConfig())
+    @property
+    def signature(self):
+        return contract_signature(self)
+
+    def __init__(
+        self,
+        config: MeshGraphNetConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__()
+        self.config = config or MeshGraphNetConfig()
+        initialize_contract(self, inputs, outputs)
         common = {
             "mlp_hidden_dim": self.config.mlp_hidden_dim,
             "mlp_layers": self.config.mlp_layers,
@@ -103,65 +119,51 @@
             **common,
         )

-    @property
-    def required_inputs(self) -> set[str]:
-        return {"node_features", "edge_index", "edge_features"}
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            node_features = inputs["node_features"]
+            edge_features = inputs["edge_features"]
+            edge_index = inputs["edge_index"]
+            if not node_features.is_floating_point():
+                raise TypeError("node_features must use a floating-point dtype.")
+            if node_features.shape[1] != self.config.node_input_dim:
+                raise ValueError(
+                    "MeshGraphNet expects node feature dimension "
+                    f"{self.config.node_input_dim}, got {node_features.shape[1]}."
+                )
+            if edge_features.dim() != 2:
+                raise ValueError(
+                    f"edge_features must have shape [E, F_edge], got {tuple(edge_features.shape)}."
+                )
+            if not edge_features.is_floating_point():
+                raise TypeError("edge_features must use a floating-point dtype.")
+            if edge_features.shape[0] != edge_index.shape[1]:
+                raise ValueError(
+                    "edge_features and edge_index must describe the same number of edges: "
+                    f"got {edge_features.shape[0]} and {edge_index.shape[1]}."
+                )
+            if edge_features.shape[1] != self.config.edge_input_dim:
+                raise ValueError(
+                    "MeshGraphNet expects edge feature dimension "
+                    f"{self.config.edge_input_dim}, got {edge_features.shape[1]}."
+                )
+            parameter = next(self.parameters())
+            node_features = node_features.to(device=parameter.device, dtype=parameter.dtype)
+            edge_features = edge_features.to(device=parameter.device, dtype=parameter.dtype)
+            edge_index = edge_index.to(device=parameter.device, dtype=torch.long)

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        node_features = self._ensure_tensor(inputs.node_features, name="node_features")
-        edge_features = self._ensure_tensor(inputs.edge_features, name="edge_features")
-        edge_index = self._ensure_tensor(inputs.edge_index, name="edge_index")
-
-        if not node_features.is_floating_point():
-            raise TypeError("node_features must use a floating-point dtype.")
-        if node_features.shape[1] != self.config.node_input_dim:
-            raise ValueError(
-                "MeshGraphNet expects node feature dimension "
-                f"{self.config.node_input_dim}, got {node_features.shape[1]}."
+            node_latent, edge_latent = self.encoder(node_features, edge_features)
+            node_latent, edge_latent = self.processor(
+                node_latent, edge_latent, edge_index
             )
-        if edge_features.dim() != 2:
-            raise ValueError(
-                f"edge_features must have shape [E, F_edge], got {tuple(edge_features.shape)}."
-            )
-        if not edge_features.is_floating_point():
-            raise TypeError("edge_features must use a floating-point dtype.")
-        if edge_features.shape[0] != edge_index.shape[1]:
-            raise ValueError(
-                "edge_features and edge_index must describe the same number of edges: "
-                f"got {edge_features.shape[0]} and {edge_index.shape[1]}."
-            )
-        if edge_features.shape[1] != self.config.edge_input_dim:
-            raise ValueError(
-                "MeshGraphNet expects edge feature dimension "
-                f"{self.config.edge_input_dim}, got {edge_features.shape[1]}."
-            )
-        if edge_index.dtype == torch.bool or edge_index.is_floating_point() or edge_index.is_complex():
-            raise TypeError("edge_index must use an integer dtype.")
-        return True
-
-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        node_features = self._ensure_tensor(
-            inputs.node_features, name="node_features"
-        ).to(device=parameter.device, dtype=parameter.dtype)
-        edge_features = self._ensure_tensor(
-            inputs.edge_features, name="edge_features"
-        ).to(device=parameter.device, dtype=parameter.dtype)
-        edge_index = torch.as_tensor(
-            inputs.edge_index, device=parameter.device, dtype=torch.long
-        )
-
-        node_latent, edge_latent = self.encoder(node_features, edge_features)
-        node_latent, edge_latent = self.processor(
-            node_latent, edge_latent, edge_index
-        )
-        predictions = self.decoder(node_latent)
-        metadata = {"batch": inputs.batch} if inputs.batch is not None else {}
-        return ModelOutput(
-            predictions=predictions,
-            latent=node_latent,
-            auxiliary={"edge_latent": edge_latent},
-            metadata=metadata,
-        )
+            predictions = self.decoder(node_latent)
+            metadata = {"batch": inputs["batch"]} if inputs.get("batch") is not None else {}
+            results.append({
+                "predictions": predictions,
+                "latent": node_latent,
+                "auxiliary": {"edge_latent": edge_latent},
+                "metadata": metadata,
+            })
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>pointnet/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet/__init__.py
+++ src/PINNCore/networks/pointnet/__init__.py
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
+++ src/PINNCore/networks/pointnet/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from PINNCore.networks.normalization import point_normalization


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
+++ src/PINNCore/networks/pointnet/model.py
@@ -1,12 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
+from PINNCore.networks.normalization import normalization_mask, masked_point_inputs

 import torch

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.checks import check
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig

 from .components import (
     PointNetClassificationHead,
@@ -43,8 +48,8 @@
             raise ValueError("global_dim must be positive.")
         if self.task not in {"classification", "segmentation"}:
             raise ValueError("task must be 'classification' or 'segmentation'.")
-        if self.normalization not in {"batch", "none"}:
-            raise ValueError("normalization must be batch or none.")
+        if self.normalization not in {"batch", "instance", "none"}:
+            raise ValueError("normalization must be batch, instance, or none.")
         if not 0.0 <= self.dropout < 1.0:
             raise ValueError("dropout must be in the range [0, 1).")

@@ -53,13 +58,22 @@
         return self.input_dim - 3


-class PointNet(BaseNetwork):
+class PointNet(Model):
     """PointNet for global classification and point-wise prediction."""

     config_class = PointNetConfig
+    vmap_randomness = "different"

-    def __init__(self, config: PointNetConfig | None = None):
-        super().__init__(config=config or PointNetConfig())
+    def __init__(
+        self,
+        config: PointNetConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__()
+        self.config = config or PointNetConfig()
+        initialize_contract(self, inputs, outputs)
         self.encoder = PointNetEncoder(
             self.config.input_dim,
             self.config.global_dim,
@@ -83,88 +97,78 @@
             )

     @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates"}
+    def signature(self):
+        return contract_signature(self)

-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="POINT_CLOUD",
-            requires={"coordinates"},
-            supports={"point_features", "point_mask"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            points = inputs["coordinates"]
+            if points.ndim != 3 or points.shape[-1] != 3:
+                raise ValueError(
+                    "PointNet coordinates must have shape [B, N, 3], "
+                    f"got {tuple(points.shape)}."
+                )
+            if not points.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            if points.shape[1] < 1:
+                raise ValueError("Each point cloud must contain at least one point.")

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        points = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        if points.ndim != 3 or points.shape[-1] != 3:
-            raise ValueError(
-                "PointNet coordinates must have shape [B, N, 3], "
-                f"got {tuple(points.shape)}."
+            expected_features = self.config.feature_dim
+            if inputs.get("point_features") is None:
+                if expected_features:
+                    raise ValueError(
+                        f"PointNet expects {expected_features} additional point features."
+                    )
+            else:
+                features = inputs.get("point_features")
+                expected_shape = (*points.shape[:2], expected_features)
+                if tuple(features.shape) != expected_shape:
+                    raise ValueError(
+                        f"point_features must have shape {expected_shape}, "
+                        f"got {tuple(features.shape)}."
+                    )
+                if not features.is_floating_point():
+                    raise TypeError("point_features must use a floating-point dtype.")
+
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"))
+                if tuple(mask.shape) != tuple(points.shape[:2]):
+                    raise ValueError(
+                        f"point_mask must have shape {tuple(points.shape[:2])}, "
+                        f"got {tuple(mask.shape)}."
+                    )
+                if mask.dtype != torch.bool:
+                    raise TypeError("point_mask must use a boolean dtype.")
+                check(mask.any(dim=1), "Every point cloud must contain at least one valid point.")
+
+            parameter = next(self.parameters())
+            points = torch.as_tensor(
+                inputs["coordinates"], device=parameter.device, dtype=parameter.dtype
             )
-        if not points.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        if points.shape[1] < 1:
-            raise ValueError("Each point cloud must contain at least one point.")
+            features = None
+            if inputs.get("point_features") is not None:
+                features = torch.as_tensor(
+                    inputs.get("point_features"), device=parameter.device, dtype=parameter.dtype
+                )
+            mask = None
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"), device=parameter.device, dtype=torch.bool)

-        expected_features = self.config.feature_dim
-        if inputs.point_features is None:
-            if expected_features:
-                raise ValueError(
-                    f"PointNet expects {expected_features} additional point features."
-                )
-        else:
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            expected_shape = (*points.shape[:2], expected_features)
-            if tuple(features.shape) != expected_shape:
-                raise ValueError(
-                    f"point_features must have shape {expected_shape}, "
-                    f"got {tuple(features.shape)}."
-                )
-            if not features.is_floating_point():
-                raise TypeError("point_features must use a floating-point dtype.")
-
-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask)
-            if tuple(mask.shape) != tuple(points.shape[:2]):
-                raise ValueError(
-                    f"point_mask must have shape {tuple(points.shape[:2])}, "
-                    f"got {tuple(mask.shape)}."
-                )
-            if mask.dtype != torch.bool:
-                raise TypeError("point_mask must use a boolean dtype.")
-            if not torch.all(mask.any(dim=1)):
-                raise ValueError("Every point cloud must contain at least one valid point.")
-        return True
-
-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        points = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        features = None
-        if inputs.point_features is not None:
-            features = torch.as_tensor(
-                inputs.point_features, device=parameter.device, dtype=parameter.dtype
-            )
-        mask = None
-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
-
-        local, global_feature, transforms = self.encoder(points, features, mask)
-        if self.config.task == "classification":
-            predictions = self.head(global_feature)
-        else:
-            predictions = self.head(local, global_feature)
-            if mask is not None:
-                predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
-
-        return ModelOutput(
-            predictions=predictions,
-            latent=global_feature,
-            auxiliary=transforms,
-            metadata={"point_mask": mask} if mask is not None else {},
-        )
+            points, features = masked_point_inputs(points, features, mask)
+            with normalization_mask(mask, enabled=mask is not None):
+                local, global_feature, transforms = self.encoder(points, features, mask)
+                if self.config.task == "classification":
+                    predictions = self.head(global_feature)
+                else:
+                    predictions = self.head(local, global_feature)
+                    if mask is not None:
+                        predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
+            results.append({
+                "predictions": predictions,
+                "latent": global_feature,
+                "auxiliary": transforms,
+                "metadata": {"point_mask": mask} if mask is not None else {},
+            })
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>pointnet2/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/pointnet2/__init__.py
+++ src/PINNCore/networks/pointnet2/__init__.py
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
+++ src/PINNCore/networks/pointnet2/components.py
@@ -4,6 +4,8 @@

 import torch
 from torch import nn
+from PINNCore.checks import check
+from PINNCore.networks.normalization import BatchNorm2d, normalization_mask, has_normalization_mask


 def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
@@ -43,22 +45,21 @@
             (batch_size, point_count), dtype=torch.bool, device=points.device
         )
     valid_counts = mask.sum(dim=1)
-    if torch.any(valid_counts == 0):
-        raise ValueError("Every point cloud must contain at least one valid point.")
+    check(valid_counts > 0, "Every point cloud must contain at least one valid point.")

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
@@ -68,7 +69,7 @@
         torch.arange(sample_count, device=points.device).unsqueeze(0)
         < valid_counts.unsqueeze(1).clamp_max(sample_count)
     )
-    return indices, sampled_mask
+    return torch.stack(indices, dim=1), sampled_mask


 def query_ball_point(
@@ -145,7 +146,7 @@
                 )
             )
             if normalization == "batch":
-                layers.append(nn.BatchNorm2d(output_channels))
+                layers.append(BatchNorm2d(output_channels))
             layers.append(nn.ReLU(inplace=True))
         self.layers = nn.Sequential(*layers)

@@ -218,7 +219,12 @@
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
@@ -239,7 +245,7 @@
         normalization: str = "batch",
     ):
         super().__init__()
-        from models.pointnet.components import SharedPointMLP
+        from ..pointnet.components import SharedPointMLP

         self.mlp = SharedPointMLP(
             (input_channels, *mlp_channels), normalization=normalization
@@ -269,7 +275,8 @@
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
+++ src/PINNCore/networks/pointnet2/model.py
@@ -1,13 +1,18 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping
+
+from PINNCore.quantities import Quantity
+from PINNCore.networks.normalization import normalization_mask, masked_point_inputs

 import torch

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.pointnet.components import PointNetClassificationHead
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.checks import check
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig
+from ..pointnet.components import PointNetClassificationHead

 from .components import PointNet2Encoder, PointNet2SegmentationHead

@@ -76,13 +81,22 @@
         return self.input_dim - 3


-class PointNet2(BaseNetwork):
+class PointNet2(Model):
     """PointNet++ with hierarchical local abstraction and feature propagation."""

     config_class = PointNet2Config
-
-    def __init__(self, config: PointNet2Config | None = None):
-        super().__init__(config=config or PointNet2Config())
+    vmap_randomness = "different"
+
+    def __init__(
+        self,
+        config: PointNet2Config | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__()
+        self.config = config or PointNet2Config()
+        initialize_contract(self, inputs, outputs)
         self.encoder = PointNet2Encoder(
             self.config.feature_dim,
             self.config.sample_counts,
@@ -111,97 +125,88 @@
             )

     @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates"}
-
-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="POINT_CLOUD",
-            requires={"coordinates"},
-            supports={"point_features", "point_mask"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
-
-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        points = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        if points.ndim != 3 or points.shape[-1] != 3:
-            raise ValueError(
-                "PointNet++ coordinates must have shape [B, N, 3], "
-                f"got {tuple(points.shape)}."
-            )
-        if not points.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        if points.shape[1] < 1:
-            raise ValueError("Each point cloud must contain at least one point.")
-
-        if inputs.point_features is None:
-            if self.config.feature_dim:
+    def signature(self):
+        return contract_signature(self)
+
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            points = inputs["coordinates"]
+            if points.ndim != 3 or points.shape[-1] != 3:
                 raise ValueError(
-                    f"PointNet++ expects {self.config.feature_dim} additional point features."
+                    "PointNet++ coordinates must have shape [B, N, 3], "
+                    f"got {tuple(points.shape)}."
                 )
-        else:
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            expected = (*points.shape[:2], self.config.feature_dim)
-            if tuple(features.shape) != expected:
-                raise ValueError(
-                    f"point_features must have shape {expected}, got {tuple(features.shape)}."
+            if not points.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            if points.shape[1] < 1:
+                raise ValueError("Each point cloud must contain at least one point.")
+
+            if inputs.get("point_features") is None:
+                if self.config.feature_dim:
+                    raise ValueError(
+                        f"PointNet++ expects {self.config.feature_dim} additional point features."
+                    )
+            else:
+                features = inputs.get("point_features")
+                expected = (*points.shape[:2], self.config.feature_dim)
+                if tuple(features.shape) != expected:
+                    raise ValueError(
+                        f"point_features must have shape {expected}, got {tuple(features.shape)}."
+                    )
+                if not features.is_floating_point():
+                    raise TypeError("point_features must use a floating-point dtype.")
+
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"))
+                if tuple(mask.shape) != tuple(points.shape[:2]):
+                    raise ValueError(
+                        f"point_mask must have shape {tuple(points.shape[:2])}, "
+                        f"got {tuple(mask.shape)}."
+                    )
+                if mask.dtype != torch.bool:
+                    raise TypeError("point_mask must use a boolean dtype.")
+                check(mask.any(dim=1), "Every point cloud must contain at least one valid point.")
+
+            parameter = next(self.parameters())
+            points = torch.as_tensor(
+                inputs["coordinates"], device=parameter.device, dtype=parameter.dtype
+            )
+            features = None
+            if inputs.get("point_features") is not None:
+                features = torch.as_tensor(
+                    inputs.get("point_features"), device=parameter.device, dtype=parameter.dtype
                 )
-            if not features.is_floating_point():
-                raise TypeError("point_features must use a floating-point dtype.")
-
-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask)
-            if tuple(mask.shape) != tuple(points.shape[:2]):
-                raise ValueError(
-                    f"point_mask must have shape {tuple(points.shape[:2])}, "
-                    f"got {tuple(mask.shape)}."
+            mask = (
+                torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
+                if inputs.get("point_mask") is None
+                else torch.as_tensor(inputs.get("point_mask"), device=points.device, dtype=torch.bool)
+            )
+
+            points, features = masked_point_inputs(points, features, mask if inputs.get("point_mask") is not None else None)
+            with normalization_mask(mask, enabled=inputs.get("point_mask") is not None):
+                global_feature, level_points, level_features, level_masks = self.encoder(
+                    points, features, mask
                 )
-            if mask.dtype != torch.bool:
-                raise TypeError("point_mask must use a boolean dtype.")
-            if not torch.all(mask.any(dim=1)):
-                raise ValueError("Every point cloud must contain at least one valid point.")
-        return True
-
-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        points = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        features = None
-        if inputs.point_features is not None:
-            features = torch.as_tensor(
-                inputs.point_features, device=parameter.device, dtype=parameter.dtype
-            )
-        mask = (
-            torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
-            if inputs.point_mask is None
-            else torch.as_tensor(inputs.point_mask, device=points.device, dtype=torch.bool)
-        )
-
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
-        return ModelOutput(
-            predictions=predictions,
-            latent=global_feature,
-            auxiliary={
-                "sampled_coordinates": level_points,
-                "sampled_masks": level_masks,
-            },
-            metadata={"point_mask": mask},
-        )
+                if self.config.task == "classification":
+                    predictions = self.head(global_feature)
+                else:
+                    predictions = self.head(
+                        points,
+                        features,
+                        mask,
+                        level_points,
+                        level_features,
+                        level_masks,
+                    )
+            results.append({
+                "predictions": predictions,
+                "latent": global_feature,
+                "auxiliary": {
+                    "sampled_coordinates": level_points,
+                    "sampled_masks": level_masks,
+                },
+                "metadata": {"point_mask": mask},
+            })
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>registry.py</summary>

```diff
--- third_party/neural3dbench/models/registry.py
+++ src/PINNCore/networks/registry.py
@@ -1,20 +1,22 @@
 from __future__ import annotations

 import inspect
-from typing import Any, get_args, get_origin, Union
+from typing import Any, Mapping, get_args, get_origin, Union

-from .base import BaseNetwork
+from PINNCore.quantities import Quantity
+
+from PINNCore.model import Model
 from .configs import BaseModelConfig


 class ModelRegistry:
-    _registry: dict[str, type[BaseNetwork]] = {}
+    _registry: dict[str, type[Model]] = {}
     _config_types: dict[str, type[BaseModelConfig]] = {}

     @classmethod
-    def register(cls, name: str, model_class: type[BaseNetwork], config_class: type[BaseModelConfig] | None = None) -> None:
-        if not isinstance(model_class, type) or not issubclass(model_class, BaseNetwork):
-            raise TypeError("Registered model must be a subclass of BaseNetwork.")
+    def register(cls, name: str, model_class: type[Model], config_class: type[BaseModelConfig] | None = None) -> None:
+        if not isinstance(model_class, type) or not issubclass(model_class, Model):
+            raise TypeError("Registered model must be a subclass of Model.")
         if name in cls._registry:
             raise ValueError(f"Model '{name}' is already registered.")

@@ -25,7 +27,7 @@
         cls._config_types[name] = config_class

     @classmethod
-    def get(cls, name: str) -> type[BaseNetwork]:
+    def get(cls, name: str) -> type[Model]:
         try:
             return cls._registry[name]
         except KeyError as exc:
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
+    ) -> Model:
         model_class = cls.get(name)
         config_type = cls.get_config_class(name)

@@ -49,10 +59,10 @@
         elif config is not None and not isinstance(config, BaseModelConfig):
             raise TypeError("Model config must be a BaseModelConfig or dict.")

-        return model_class(config=config)
+        return model_class(config=config, inputs=inputs, outputs=outputs)

     @classmethod
-    def _infer_config_class(cls, model_class: type[BaseNetwork]) -> type[BaseModelConfig]:
+    def _infer_config_class(cls, model_class: type[Model]) -> type[BaseModelConfig]:
         try:
             signature = inspect.signature(model_class.__init__)
         except (TypeError, ValueError):
@@ -78,8 +88,8 @@
         return BaseModelConfig


-def create_model(config: dict[str, Any] | BaseModelConfig | str | None = None, **kwargs: Any) -> BaseNetwork:
-    if isinstance(config, BaseNetwork):
+def create_model(config: dict[str, Any] | BaseModelConfig | str | None = None, **kwargs: Any) -> Model:
+    if isinstance(config, Model):
         return config
     if isinstance(config, str):
         return ModelRegistry.create(config, **kwargs)
```

</details>

<details>
<summary>siren/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/siren/__init__.py
+++ src/PINNCore/networks/siren/__init__.py
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
+++ src/PINNCore/networks/siren/model.py
@@ -2,13 +2,17 @@

 import math
 from dataclasses import dataclass
+from typing import Mapping

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.quantities import Quantity
+from PINNCore.checks import check
+
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig

 from .layers import SineLayer

@@ -44,13 +48,22 @@
             raise TypeError("outermost_linear must be boolean.")


-class SIREN(BaseNetwork):
+class SIREN(Model):
     """Pointwise sinusoidal representation network for continuous fields."""

     config_class = SIRENConfig
+    vmap_randomness = "different"

-    def __init__(self, config: SIRENConfig | None = None) -> None:
-        super().__init__(config=config or SIRENConfig())
+    def __init__(
+        self,
+        config: SIRENConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__()
+        self.config = config or SIRENConfig()
+        initialize_contract(self, inputs, outputs)
         total_input_dim = self.config.spatial_dim + self.config.input_dim
         self.first_layer = SineLayer(
             total_input_dim,
@@ -74,90 +87,79 @@
             self.output_layer.weight.uniform_(-output_bound, output_bound)

     @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates"}
+    def signature(self):
+        return contract_signature(self)

-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="POINT_CLOUD",
-            requires={"coordinates"},
-            supports={"point_features", "point_mask"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            coordinates = inputs["coordinates"]
+            if coordinates.ndim != 3 or coordinates.shape[-1] != self.config.spatial_dim:
+                raise ValueError(
+                    f"coordinates must have shape [B, N, {self.config.spatial_dim}], "
+                    f"got {tuple(coordinates.shape)}."
+                )
+            if not coordinates.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            if coordinates.shape[0] < 1 or coordinates.shape[1] < 1:
+                raise ValueError("Every sample must contain at least one point.")

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        if coordinates.ndim != 3 or coordinates.shape[-1] != self.config.spatial_dim:
-            raise ValueError(
-                f"coordinates must have shape [B, N, {self.config.spatial_dim}], "
-                f"got {tuple(coordinates.shape)}."
+            if self.config.input_dim > 0:
+                if inputs.get("point_features") is None:
+                    raise ValueError(f"SIREN expects {self.config.input_dim} point features.")
+                features = inputs["point_features"]
+                expected = (*coordinates.shape[:2], self.config.input_dim)
+                if tuple(features.shape) != expected:
+                    raise ValueError(
+                        f"point_features must have shape {expected}, got {tuple(features.shape)}."
+                    )
+                if not features.is_floating_point():
+                    raise TypeError("point_features must use a floating-point dtype.")
+            elif inputs.get("point_features") is not None:
+                features = inputs["point_features"]
+                expected = (*coordinates.shape[:2], 0)
+                if tuple(features.shape) != expected:
+                    raise ValueError("Set input_dim to use point_features.")
+                if not features.is_floating_point():
+                    raise TypeError("point_features must use a floating-point dtype.")
+
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"))
+                expected_mask = tuple(coordinates.shape[:2])
+                if tuple(mask.shape) != expected_mask:
+                    raise ValueError(
+                        f"point_mask must have shape {expected_mask}, got {tuple(mask.shape)}."
+                    )
+                if mask.dtype != torch.bool:
+                    raise TypeError("point_mask must use a boolean dtype.")
+                check(mask.any(dim=1), "Every sample must contain at least one valid point.")
+
+            parameter = next(self.parameters())
+            coordinates = torch.as_tensor(
+                inputs.get("coordinates"), device=parameter.device, dtype=parameter.dtype
             )
-        if not coordinates.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        if coordinates.shape[0] < 1 or coordinates.shape[1] < 1:
-            raise ValueError("Every sample must contain at least one point.")
+            x = coordinates
+            if self.config.input_dim > 0:
+                features = torch.as_tensor(
+                    inputs.get("point_features"), device=parameter.device, dtype=parameter.dtype
+                )
+                x = torch.cat((coordinates, features), dim=-1)

-        if self.config.input_dim > 0:
-            if inputs.point_features is None:
-                raise ValueError(f"SIREN expects {self.config.input_dim} point features.")
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            expected = (*coordinates.shape[:2], self.config.input_dim)
-            if tuple(features.shape) != expected:
-                raise ValueError(
-                    f"point_features must have shape {expected}, got {tuple(features.shape)}."
-                )
-            if not features.is_floating_point():
-                raise TypeError("point_features must use a floating-point dtype.")
-        elif inputs.point_features is not None:
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            expected = (*coordinates.shape[:2], 0)
-            if tuple(features.shape) != expected:
-                raise ValueError("Set input_dim to use point_features.")
-            if not features.is_floating_point():
-                raise TypeError("point_features must use a floating-point dtype.")
+            x = self.first_layer(x)
+            for layer in self.hidden_layers:
+                x = layer(x)
+            predictions = self.output_layer(x)
+            if not self.config.outermost_linear:
+                predictions = torch.sin(self.config.hidden_omega_0 * predictions)

-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask)
-            expected_mask = tuple(coordinates.shape[:2])
-            if tuple(mask.shape) != expected_mask:
-                raise ValueError(
-                    f"point_mask must have shape {expected_mask}, got {tuple(mask.shape)}."
-                )
-            if mask.dtype != torch.bool:
-                raise TypeError("point_mask must use a boolean dtype.")
-            if not torch.all(mask.any(dim=1)):
-                raise ValueError("Every sample must contain at least one valid point.")
-        return True
-
-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        coordinates = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        x = coordinates
-        if self.config.input_dim > 0:
-            features = torch.as_tensor(
-                inputs.point_features, device=parameter.device, dtype=parameter.dtype
-            )
-            x = torch.cat((coordinates, features), dim=-1)
-
-        x = self.first_layer(x)
-        for layer in self.hidden_layers:
-            x = layer(x)
-        predictions = self.output_layer(x)
-        if not self.config.outermost_linear:
-            predictions = torch.sin(self.config.hidden_omega_0 * predictions)
-
-        mask = None
-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask, device=parameter.device, dtype=torch.bool)
-            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
-        return ModelOutput(
-            predictions=predictions,
-            latent=x,
-            metadata={"point_mask": mask} if mask is not None else {},
-        )
+            mask = None
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"), device=parameter.device, dtype=torch.bool)
+                predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
+            results.append(dict(
+                predictions=predictions,
+                latent=x,
+                metadata={"point_mask": mask} if mask is not None else {},
+            ))
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>transolver/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/transolver/__init__.py
+++ src/PINNCore/networks/transolver/__init__.py
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
+++ src/PINNCore/networks/transolver/model.py
@@ -1,13 +1,17 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.quantities import Quantity
+from PINNCore.checks import check
+
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig

 from .block import TransolverBlock
 from .components import InputEmbedding, OutputHead
@@ -49,13 +53,22 @@
             raise ValueError("activation must be gelu, relu, silu, or tanh.")


-class Transolver(BaseNetwork):
+class Transolver(Model):
     """Transolver for point-wise physical-field prediction on irregular domains."""

     config_class = TransolverConfig
+    vmap_randomness = "different"

-    def __init__(self, config: TransolverConfig | None = None) -> None:
-        super().__init__(config=config or TransolverConfig())
+    def __init__(
+        self,
+        config: TransolverConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ) -> None:
+        super().__init__()
+        self.config = config or TransolverConfig()
+        initialize_contract(self, inputs, outputs)
         self.input_embedding = InputEmbedding(
             self.config.spatial_dim,
             self.config.input_dim,
@@ -90,81 +103,70 @@
             nn.init.zeros_(module.bias)

     @property
-    def required_inputs(self) -> set[str]:
-        return {"coordinates"}
+    def signature(self):
+        return contract_signature(self)

-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="POINT_CLOUD",
-            requires={"coordinates"},
-            supports={"point_features", "point_mask"},
-            variable_geometry=True,
-            variable_node_count=True,
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            coordinates = inputs["coordinates"]
+            if coordinates.ndim != 3 or coordinates.shape[-1] != self.config.spatial_dim:
+                raise ValueError(
+                    f"coordinates must have shape [B, N, {self.config.spatial_dim}], "
+                    f"got {tuple(coordinates.shape)}."
+                )
+            if not coordinates.is_floating_point():
+                raise TypeError("coordinates must use a floating-point dtype.")
+            if coordinates.shape[1] < 1:
+                raise ValueError("Each sample must contain at least one point.")
+            if self.config.input_dim:
+                if inputs.get("point_features") is None:
+                    raise ValueError(f"Transolver expects {self.config.input_dim} point features.")
+                features = inputs["point_features"]
+                expected = (*coordinates.shape[:2], self.config.input_dim)
+                if tuple(features.shape) != expected:
+                    raise ValueError(f"point_features must have shape {expected}.")
+                if not features.is_floating_point():
+                    raise TypeError("point_features must use a floating-point dtype.")
+            elif inputs.get("point_features") is not None:
+                features = inputs["point_features"]
+                if features.shape[-1] != 0:
+                    raise ValueError("Set input_dim to use point_features.")
+            if inputs.get("point_mask") is not None:
+                mask = torch.as_tensor(inputs.get("point_mask"))
+                if tuple(mask.shape) != tuple(coordinates.shape[:2]) or mask.dtype != torch.bool:
+                    raise ValueError("point_mask must be boolean with shape [B, N].")
+                check(mask.any(dim=1), "Every sample must contain at least one valid point.")

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
-        if coordinates.ndim != 3 or coordinates.shape[-1] != self.config.spatial_dim:
-            raise ValueError(
-                f"coordinates must have shape [B, N, {self.config.spatial_dim}], "
-                f"got {tuple(coordinates.shape)}."
+            parameter = next(self.parameters())
+            coordinates = torch.as_tensor(
+                inputs.get("coordinates"), device=parameter.device, dtype=parameter.dtype
             )
-        if not coordinates.is_floating_point():
-            raise TypeError("coordinates must use a floating-point dtype.")
-        if coordinates.shape[1] < 1:
-            raise ValueError("Each sample must contain at least one point.")
-        if self.config.input_dim:
-            if inputs.point_features is None:
-                raise ValueError(f"Transolver expects {self.config.input_dim} point features.")
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            expected = (*coordinates.shape[:2], self.config.input_dim)
-            if tuple(features.shape) != expected:
-                raise ValueError(f"point_features must have shape {expected}.")
-            if not features.is_floating_point():
-                raise TypeError("point_features must use a floating-point dtype.")
-        elif inputs.point_features is not None:
-            features = self._ensure_tensor(inputs.point_features, name="point_features")
-            if features.shape[-1] != 0:
-                raise ValueError("Set input_dim to use point_features.")
-        if inputs.point_mask is not None:
-            mask = torch.as_tensor(inputs.point_mask)
-            if tuple(mask.shape) != tuple(coordinates.shape[:2]) or mask.dtype != torch.bool:
-                raise ValueError("point_mask must be boolean with shape [B, N].")
-            if not torch.all(mask.any(dim=1)):
-                raise ValueError("Every sample must contain at least one valid point.")
-        return True
+            features = None if inputs.get("point_features") is None else torch.as_tensor(
+                inputs.get("point_features"), device=parameter.device, dtype=parameter.dtype
+            )
+            mask = None if inputs.get("point_mask") is None else torch.as_tensor(
+                inputs.get("point_mask"), device=parameter.device, dtype=torch.bool
+            )
+            hidden = self.input_embedding(coordinates, features) + self.placeholder
+            if mask is not None:
+                hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        coordinates = torch.as_tensor(
-            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
-        )
-        features = None if inputs.point_features is None else torch.as_tensor(
-            inputs.point_features, device=parameter.device, dtype=parameter.dtype
-        )
-        mask = None if inputs.point_mask is None else torch.as_tensor(
-            inputs.point_mask, device=parameter.device, dtype=torch.bool
-        )
-        hidden = self.input_embedding(coordinates, features) + self.placeholder
-        if mask is not None:
-            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
-
-        attention_data = []
-        for block in self.blocks:
-            if self.config.return_attention_weights:
-                hidden, block_data = block(hidden, mask, return_attention=True)
-                attention_data.append(block_data)
-            else:
-                hidden = block(hidden, mask)
-        predictions = self.output_head(hidden)
-        if mask is not None:
-            predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
-        return ModelOutput(
-            predictions=predictions,
-            latent=hidden,
-            auxiliary={"physics_attention": attention_data} if attention_data else {},
-            metadata={"point_mask": mask} if mask is not None else {},
-        )
+            attention_data = []
+            for block in self.blocks:
+                if self.config.return_attention_weights:
+                    hidden, block_data = block(hidden, mask, return_attention=True)
+                    attention_data.append(block_data)
+                else:
+                    hidden = block(hidden, mask)
+            predictions = self.output_head(hidden)
+            if mask is not None:
+                predictions = predictions.masked_fill(~mask.unsqueeze(-1), 0.0)
+            results.append(dict(
+                predictions=predictions,
+                latent=hidden,
+                auxiliary={"physics_attention": attention_data} if attention_data else {},
+                metadata={"point_mask": mask} if mask is not None else {},
+            ))
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>unet3d/__init__.py</summary>

```diff
--- third_party/neural3dbench/models/unet3d/__init__.py
+++ src/PINNCore/networks/unet3d/__init__.py
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
+++ src/PINNCore/networks/unet3d/components.py
@@ -4,6 +4,7 @@

 import torch
 from torch import nn
+from PINNCore.networks.normalization import BatchNorm3d
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
+++ src/PINNCore/networks/unet3d/model.py
@@ -1,13 +1,16 @@
 from __future__ import annotations

 from dataclasses import dataclass
+from typing import Mapping

 import torch
 from torch import nn

-from models.base import BaseNetwork
-from models.configs import BaseModelConfig
-from models.types import ModelCapabilities, ModelInput, ModelOutput
+from PINNCore.model import Model
+from PINNCore.quantities import Quantity
+
+from .._contract import initialize_contract, contract_signature, prepare_inputs, finish_outputs
+from ..configs import BaseModelConfig

 from .components import DecoderBlock3D, EncoderBlock3D, VolumetricOutputHead

@@ -55,13 +58,22 @@
             raise ValueError("upsampling must be transpose_conv or interpolate.")


-class UNet3D(BaseNetwork):
+class UNet3D(Model):
     """Encoder-decoder 3D U-Net producing raw voxel-wise logits or values."""

     config_class = UNet3DConfig
+    vmap_randomness = "different"

-    def __init__(self, config: UNet3DConfig | None = None):
-        super().__init__(config=config or UNet3DConfig())
+    def __init__(
+        self,
+        config: UNet3DConfig | None = None,
+        *,
+        inputs: Mapping[str, Quantity],
+        outputs: Mapping[str, Quantity],
+    ):
+        super().__init__()
+        self.config = config or UNet3DConfig()
+        initialize_contract(self, inputs, outputs)
         channels = [
             self.config.base_channels * self.config.channel_multiplier**level
             for level in range(self.config.levels)
@@ -99,60 +111,50 @@
         self.head = VolumetricOutputHead(channels[0], self.config.output_channels)

     @property
-    def required_inputs(self) -> set[str]:
-        return {"voxel_fields"}
+    def signature(self):
+        return contract_signature(self)

-    @property
-    def capabilities(self) -> ModelCapabilities:
-        return ModelCapabilities(
-            representation="GRID",
-            requires={"voxel_fields"},
-            supports={"boundary_mask", "material_properties"},
-            variable_geometry=False,
-            variable_node_count=False,
-        )
+    def evaluate(self, values):
+        requests, layout = prepare_inputs(self, values)
+        results = []
+        for inputs in requests:
+            volume = inputs["voxel_fields"]
+            if volume.ndim != 5:
+                raise ValueError(
+                    "UNet3D voxel_fields must have shape [B, C, D, H, W], "
+                    f"got {tuple(volume.shape)}."
+                )
+            if volume.shape[1] != self.config.input_channels:
+                raise ValueError(
+                    f"UNet3D expects {self.config.input_channels} input channels, "
+                    f"got {volume.shape[1]}."
+                )
+            if not volume.is_floating_point():
+                raise TypeError("voxel_fields must use a floating-point dtype.")
+            minimum_size = 2 ** (self.config.levels - 1)
+            if any(size < minimum_size for size in volume.shape[-3:]):
+                raise ValueError(
+                    f"Each spatial dimension must be at least {minimum_size} for "
+                    f"{self.config.levels} levels."
+                )

-    def validate_inputs(self, inputs: ModelInput) -> bool:
-        super().validate_inputs(inputs)
-        volume = self._ensure_tensor(inputs.voxel_fields, name="voxel_fields")
-        if volume.ndim != 5:
-            raise ValueError(
-                "UNet3D voxel_fields must have shape [B, C, D, H, W], "
-                f"got {tuple(volume.shape)}."
+            parameter = next(self.parameters())
+            hidden = torch.as_tensor(
+                inputs.get("voxel_fields"), device=parameter.device, dtype=parameter.dtype
             )
-        if volume.shape[1] != self.config.input_channels:
-            raise ValueError(
-                f"UNet3D expects {self.config.input_channels} input channels, "
-                f"got {volume.shape[1]}."
-            )
-        if not volume.is_floating_point():
-            raise TypeError("voxel_fields must use a floating-point dtype.")
-        minimum_size = 2 ** (self.config.levels - 1)
-        if any(size < minimum_size for size in volume.shape[-3:]):
-            raise ValueError(
-                f"Each spatial dimension must be at least {minimum_size} for "
-                f"{self.config.levels} levels."
-            )
-        return True
+            skips: list[torch.Tensor] = []
+            for encoder in self.encoders:
+                skip, hidden = encoder(hidden)
+                skips.append(skip)

-    def forward(self, inputs: ModelInput) -> ModelOutput:
-        self.validate_inputs(inputs)
-        parameter = next(self.parameters())
-        hidden = torch.as_tensor(
-            inputs.voxel_fields, device=parameter.device, dtype=parameter.dtype
-        )
-        skips: list[torch.Tensor] = []
-        for encoder in self.encoders:
-            skip, hidden = encoder(hidden)
-            skips.append(skip)
-
-        bottleneck = skips[-1]
-        hidden = bottleneck
-        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
-            hidden = decoder(hidden, skip)
-        predictions = self.head(hidden)
-        return ModelOutput(
-            predictions=predictions,
-            latent=bottleneck,
-            metadata={"spatial_shape": tuple(predictions.shape[-3:])},
-        )
+            bottleneck = skips[-1]
+            hidden = bottleneck
+            for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
+                hidden = decoder(hidden, skip)
+            predictions = self.head(hidden)
+            results.append(dict(
+                predictions=predictions,
+                latent=bottleneck,
+                metadata={"spatial_shape": tuple(predictions.shape[-3:])},
+            ))
+        return finish_outputs(self, results, layout)
```

</details>

<details>
<summary>src/PINNCore/networks/_contract.py — افزوده</summary>

```diff
--- /dev/null
+++ src/PINNCore/networks/_contract.py
@@ -0,0 +1,286 @@
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
+from ..values import Value
+from ..checks import check
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
+
+def _native_input(model, raw):
+    """Translate engine component layouts to the original numeric layouts."""
+    raw = dict(raw)
+    if model.config.name in _GRIDS:
+        raw["voxel_fields"] = raw["voxel_fields"].movedim(-1, 1)
+    if "point_mask" in raw:
+        raw["point_mask"] = raw["point_mask"].squeeze(-1)
+    return raw
+
+
+def _outputs(model, result, batch_shape):
+    """Restore intrinsic Quantity shapes without folding sample axes into them."""
+    converted = {}
+    for role, q in model.output_quantities.items():
+        tensor = result[role]
+        if model.config.name in _GRIDS or (model.config.name == "gino" and role == "latent"):
+            tensor = tensor.movedim(1, -1)
+        converted[q] = tensor.reshape(batch_shape + tensor.shape[1:-1] + q.shape)
+    return converted
+
+
+def _pack_graphs(groups):
+    """Validate local indices before offsetting and packing disjoint graphs."""
+    nodes, edges, features, lengths = [], [], [], []
+    offset = 0
+    for _, raw in groups:
+        for i in range(raw["node_features"].shape[0]):
+            node = raw["node_features"][i]
+            edge = raw["edge_index"][i]
+            if node.ndim != 2 or node.shape[0] == 0:
+                raise ValueError("node_features must have shape [N, F] with N > 0")
+            if edge.ndim != 2 or edge.shape[-1] != 2:
+                raise ValueError("edge_index must have shape [E, 2]")
+            if edge.dtype not in (torch.int32, torch.int64):
+                raise TypeError("edge_index must have an integer dtype")
+            check(((edge >= 0) & (edge < node.shape[0])).all(),
+                  "edge_index entries must be in the range [0, N) for each graph")
+            lengths.append(node.shape[0])
+            nodes.append(node)
+            edges.append(edge + offset)
+            if "edge_features" in raw:
+                features.append(raw["edge_features"][i])
+            offset += node.shape[0]
+    packed = {"node_features": torch.cat(nodes), "edge_index": torch.cat(edges).T.long()}
+    if features:
+        packed["edge_features"] = torch.cat(features)
+    return packed, lengths
+
+
+def prepare_inputs(model, values):
+    """Prepare numeric requests; the calling model owns all network computation."""
+    if not model.signature.has_ellipsis:
+        specs, _, _ = _specs(model)
+        raw = {role: values[q].reshape((1,) + values[q].shape[:len(specs[role][0])] + (specs[role][1],))
+               for role, q in model.input_quantities.items()}
+        return [_native_input(model, raw)], None
+
+    batch, list_axis, groups = _prepared_batches(model, values)
+    layout = {"batch": batch, "list_axis": list_axis, "groups": groups}
+    if model.config.name in _GRAPHS:
+        packed, lengths = _pack_graphs(groups)
+        layout["lengths"] = lengths
+        return [packed], layout
+
+    shapes = {tuple(t.shape[1:] for t in raw.values()) for _, raw in groups}
+    if len(shapes) > 1 and uses_batch_statistics(model):
+        raise ValueError("Ragged grid execution requires network-specific handling before BatchNorm")
+    layout["combined"] = len(shapes) == 1
+    if layout["combined"]:
+        raw = {role: torch.cat([g[role] for _, g in groups], dim=0) for role in groups[0][1]}
+        return [_native_input(model, raw)], layout
+    return [_native_input(model, raw) for _, raw in groups], layout
+
+
+def finish_outputs(model, results, layout):
+    """Restore named outputs after the model has evaluated its numeric requests."""
+    if layout is None:
+        return _outputs(model, results[0], ())
+    groups, batch, list_axis = layout["groups"], layout["batch"], layout["list_axis"]
+    if "lengths" in layout:
+        outputs = {}
+        for role, q in model.output_quantities.items():
+            split = results[0][role].split(layout["lengths"])
+            collected, cursor = [], 0
+            for shape, raw in groups:
+                count = raw["node_features"].shape[0]
+                tensors = split[cursor:cursor + count]
+                cursor += count
+                collected.append(torch.stack(tensors).reshape(shape + (tensors[0].shape[0],) + q.shape))
+            outputs[q] = Value(collected if list_axis else collected[0], batch + ("vertex",))
+        return outputs
+
+    outputs = {q: [] for q in model.outputs}
+    if layout["combined"]:
+        flat = _outputs(model, results[0], (sum(prod(shape) for shape, _ in groups),))
+        cursor = 0
+        for shape, _ in groups:
+            count = prod(shape)
+            for q, tensor in flat.items():
+                outputs[q].append(tensor[cursor:cursor + count].reshape(shape + tensor.shape[1:]))
+            cursor += count
+    else:
+        for (shape, _), result in zip(groups, results):
+            for q, tensor in _outputs(model, result, shape).items():
+                outputs[q].append(tensor)
+    specs = _specs(model)[1]
+    return {q: Value(outputs[q] if list_axis else outputs[q][0], batch + specs[role][0])
+            for role, q in model.output_quantities.items()}
```

</details>

<details>
<summary>src/PINNCore/networks/normalization.py — افزوده</summary>

```diff
--- /dev/null
+++ src/PINNCore/networks/normalization.py
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

<details>
<summary>src/PINNCore/checks.py — افزوده</summary>

```diff
--- /dev/null
+++ src/PINNCore/checks.py
@@ -0,0 +1,38 @@
+"""Value-dependent checks for eager evaluation, including automatic vmap."""
+
+import torch
+
+
+class _Check(torch.autograd.Function):
+    @staticmethod
+    def forward(condition, message):
+        if not condition.all():
+            raise ValueError(message)
+        # A Tensor output lets vmap preserve the condition's batch dimension.
+        return condition.clone()
+
+    @staticmethod
+    def setup_context(ctx, inputs, output):
+        # Boolean conditions carry no gradients or saved state.
+        pass
+
+    @staticmethod
+    def vmap(info, in_dims, condition, message):
+        # vmap exposes its batch axis here. Applying the operation again also
+        # handles any remaining outer vmap levels before inspecting values.
+        return _Check.apply(condition, message), in_dims[0]
+
+
+def check(condition: torch.Tensor, message: str) -> None:
+    """Raise ValueError unless every element of a boolean Tensor is True.
+
+    May be called inside Model.evaluate without consuming extra batch axes.
+    If any mapped sample fails, the complete call raises with the given message.
+    This eager check reads Tensor values; torch.compile is not guaranteed and
+    checking a CUDA condition requires synchronization with the host.
+    """
+    if not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool:
+        raise TypeError("check expects a boolean Tensor")
+    if not isinstance(message, str):
+        raise TypeError("check message must be a string")
+    _Check.apply(condition, message)
```

</details>
