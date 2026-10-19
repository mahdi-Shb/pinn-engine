# بررسی محاسبات تکراری و اضافی PINNlab

> یادداشت تاریخیِ طراحی یا بررسی PINNlab قدیمی است. همهٔ پیشنهادها قرارداد فعلی نیستند؛ [رفتار اجرایی فعلی](../contracts.fa.md) را جدا ببینید. مسیرهای legacy به کد خارج از این ریپو اشاره دارند.

این گزارش بر اساس کد محلی پروژه تهیه شده است. ۱۵۲ فایل Python در pinnlab، src، examples و ابزارهای تحلیل پروژه از نظر ساختار بررسی شدند؛ مسیرهای اجرایی مرتبط با آموزش، مشتق‌گیری، هندسه، نمونه‌برداری، مرجع و UI به‌صورت هدفمند دنبال شدند. این بررسی static همراه با آزمایش شمارش فراخوانی است؛ تضمین ریاضیِ نبودن هیچ تکرار دیگر در تمام ترکیب‌های ممکن نیست. درصد افزایش سرعت اندازه‌گیری نشده است.

در این بررسی کد اجرایی کتابخانه تغییر نکرده است. «قطعی» یعنی در مسیر ذکرشده اجرای اضافه از کد یا شمارش فراخوانی روشن است؛ «وابسته به ترکیب» یعنی وجود آن به پیکربندی پروژه بستگی دارد؛ «فرصت» نیازمند مقایسهٔ هزینه و حفظ رفتار عددی است.

واژه‌ها: forward یعنی یک بار اجرای شبکه برای پیش‌بینی؛ batch یعنی مجموعهٔ نقاط و داده‌های همراه؛ scenario یعنی یک مجموعهٔ شرایط فیزیکی، مثل amplitude=1.25؛ cache یعنی نگهداری نتیجه برای استفادهٔ دوباره؛ normal یعنی جهت عمود بر سطح؛ graph یعنی مسیر محاسباتی لازم برای مشتق‌گیری؛ component یعنی یکی از خروجی‌های مدل، مثل فشار یا دما.

آزمایش‌های شمارش روی CPU انجام شدند و همگی گذشتند: محاسبهٔ normal موجود و ثابت، کپی دوبارهٔ batch ترکیبی، sampler دادهٔ مشترک، اشتراک موفق AutoGrad در sampler ثابت، FDM، Spectral، callback ضریب Robin، metrics، فاصله با تغییر شرایط، مسیر FiLM، provider ثابت، مشتق‌های NS/Energy و CG با RHS صفر. این‌ها micro-probe هستند و جای benchmark همهٔ پروژه‌ها را نمی‌گیرند.

## 1. normal موجود دوباره محاسبه و دور انداخته می‌شود

**وضعیت:** قطعی. **محل:** pinnlab/sampling/ensemble.py:122 (legacy PINNlab: `pinnlab/sampling/ensemble.py:122`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** در fields.setdefault(name, geometry.normal(points))، پایتون ابتدا normal را حساب می‌کند؛ سپس بررسی می‌کند کلید وجود دارد یا نه. اگر sampler داخلی normal را قبلاً داده باشد، نتیجهٔ محاسبهٔ جدید اصلاً مصرف نمی‌شود.

**چه زمانی رخ می‌دهد؟** برای OffsetBoundarySampler یا ClippedSurfaceSampler زیر GeometryBatchSampler، و هر sampler دارای normal.

**راه حذف یا کاهش و محدودیت آن:** قبل از فراخوانی normal وجود کلید بررسی شود. این حذف، مقدار خروجی را عوض نمی‌کند.

## 2. normal بدون نیاز مصرف‌کننده ساخته می‌شود

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/sampling/ensemble.py:108 (legacy PINNlab: `pinnlab/sampling/ensemble.py:108`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** GeometryBatchSampler برای هر هندسهٔ مرزی دارای normal آن را می‌سازد؛ این تصمیم از نوع constraint مستقل است. Dirichlet، DataFit و InterfaceValue خودشان به جهت عمود نیاز ندارند. ClippedSurfaceSampler هم همیشه normal را ضمیمه می‌کند.

**چه زمانی رخ می‌دهد؟** وقتی تنها مصرف‌کنندهٔ batch یک شرط مقدار، نه شرط شار، باشد.

**راه حذف یا کاهش و محدودیت آن:** نیاز به normal به‌صورت صریح در assembly یا sampler اعلام شود. حذف سراسری درست نیست: Robin، Neumann و surface_force به آن نیاز دارند.

## 3. normal نقاط ثابت در هر فراخوانی دوباره حساب می‌شود

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/constraints/base.py:23 (legacy PINNlab: `pinnlab/constraints/base.py:23`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** resolve_normal در نبود field:normal، هندسه را هر بار صدا می‌زند. GeometryBatchSampler._tag نیز خروجی normal را cache نمی‌کند. دو constraint روی همان سطح، یا مرحلهٔ بعد با همان مختصات، می‌توانند همان محاسبه را تکرار کنند. برای هندسهٔ دارای normal پیش‌فرض، این کار شامل مشتق‌گیری از SDF است.

**چه زمانی رخ می‌دهد؟** نقاط و هندسه ثابت‌اند، ولی شرط مرزی در هر مرحله اجرا می‌شود؛ یا فقط شرایط amplitude تغییر می‌کنند.

**راه حذف یا کاهش و محدودیت آن:** normal هندسهٔ ثابت با هویت مختصات و هندسه cache شود یا یک‌بار به batch فیزیکی ضمیمه شود. برای هندسهٔ متحرک و نقاط جدید باید invalidate شود.

## 4. چسباندن دوبارهٔ batchهای ثابت

**وضعیت:** قطعی. **محل:** pinnlab/sampling/ensemble.py:125 (legacy PINNlab: `pinnlab/sampling/ensemble.py:125`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** GeometryBatchSampler و MixtureSampler با ثابت‌ماندن batch_id همچنان torch.cat برای points، weights و fields اجرا می‌کنند. حتی یک child هم به‌وسیلهٔ cat کپی می‌شود. cache شناسه وجود دارد، ولی cache خودِ batch ترکیبی وجود ندارد.

**چه زمانی رخ می‌دهد؟** childها ثابت‌اند، یا چند term یک sampler ترکیبی را در یک step صدا می‌زنند.

**راه حذف یا کاهش و محدودیت آن:** batch ساخته‌شده با کلید child batch_idها و active memberها نگهداری شود. منبع دوم: pinnlab/sampling/mixture.py:48.

## 5. wrapperهای rotation و force target خروجی ثابت را تازه اعلام می‌کنند

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/sampling/wrappers.py:200 (legacy PINNlab: `pinnlab/sampling/wrappers.py:200`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** AoARotatedSampler هر بار sin/cos و چرخش را انجام می‌دهد و batch_id تازه می‌سازد. AirfoilForceTargetSampler هم interpolation هدف‌های نیرو را دوباره انجام می‌دهد و شناسهٔ تازه می‌دهد. اگر batch داخلی و شرایط عوض نشده باشند، خروجی یکسان است؛ شناسهٔ تازه اشتراک AutoGrad را هم از بین می‌برد.

**چه زمانی رخ می‌دهد؟** چند مصرف‌کننده در یک step یا شرایط و نقاط ثابت.

**راه حذف یا کاهش و محدودیت آن:** cache با کلید batch داخلی و تنظیمات مؤثر؛ interpolation نیرو ابتدا برای زوج‌های یکتای هندسه و شرایط و سپس broadcast شود. منبع دوم: pinnlab/sampling/forces.py:sample.

## 6. sampler داده در یک step چند بار minibatch می‌کشد

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/sampling/wrappers.py:70 (legacy PINNlab: `pinnlab/sampling/wrappers.py:70`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** DataSampler در حالت minibatch و StratifiedDataSampler در هر sample(step) دوباره ردیف انتخاب می‌کنند؛ step کلید cache نیست. بنابراین سه DataFit با یک sampler الزاماً یک minibatch مشترک ندارند و forward قابل اشتراک نیست. AirfoilFieldDataSampler حتی در full-batch هم indexing و batch_id تازه می‌سازد.

**چه زمانی رخ می‌دهد؟** یک sampler داده به چند term وصل شده است.

**راه حذف یا کاهش و محدودیت آن:** اگر هدف minibatch مشترک است، draw یک بار در هر step انجام شود. اگر draw مستقل عمدی است، این حذف رفتار آماری آموزش را عوض می‌کند. منابع دوم: wrappers.py:365 و sampling/gt_data.py:sample.

## 7. شرایط ثابت provider باعث بازسازی batch می‌شوند

**وضعیت:** قطعی در این مسیر. **محل:** pinnlab/conditioning.py:905 (legacy PINNlab: `pinnlab/conditioning.py:905`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** در مسیر provider، scenario_refresh از تغییر شمارهٔ step محاسبه می‌شود، نه از تغییر جدول شرایط provider. حتی KeyedConditionProvider(refresh_every=0) شرایط ثابت می‌دهد، ولی ConditionalSampler در step بعد repeat_interleave، fields و batch_id جدید می‌سازد.

**چه زمانی رخ می‌دهد؟** provider مشترک با refresh_every=0 یا دورهٔ refresh بزرگ‌تر از یک.

**راه حذف یا کاهش و محدودیت آن:** نسخه یا bucket واقعی جدول provider مبنای تغییر باشد. این مورد با اجرای دو step و شرایط کاملاً برابر تأیید شد.

## 8. تغییر شرایط، فاصلهٔ هندسیِ ثابت را هم invalidate می‌کند

**وضعیت:** وابسته به ترتیب wrapperها. **محل:** pinnlab/sampling/fields.py:67 (legacy PINNlab: `pinnlab/sampling/fields.py:67`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** FieldAttachingSampler تغییر batch_id داخلی را دلیل بازسازی همهٔ fields می‌داند. اگر WallDistanceFieldSampler بیرون ConditionalSampler باشد، تغییر amplitude با نقاط کاملاً ثابت هم wall.distance(points) را دوباره اجرا می‌کند؛ فاصله به amplitude وابسته نیست.

**چه زمانی رخ می‌دهد؟** هندسه ثابت است و condition refresh می‌شود.

**راه حذف یا کاهش و محدودیت آن:** برای فیلدهای مستقل از شرایط، cache مختصات جدا از cache شرایط باشد؛ در صورت معتبر بودن، wall_distance قبل از conditioning محاسبه شود. چرخش هندسه یا تغییر شکل نیازمند بررسی مستقل است.

## 9. PhysicsFieldSampler هم بخش ثابت و هم خروجی‌های اضافی را محاسبه می‌کند

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/sampling/fields.py:165 (legacy PINNlab: `pinnlab/sampling/fields.py:165`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** flow_features، فاصله، arc_length و نسبت edge_speed فقط به نقاط و panel ثابت وابسته‌اند؛ با تغییر u_inf، دوباره محاسبه می‌شوند، با اینکه بخش وابسته به سرعت را می‌توان جدا کرد. این sampler همیشه u، v، wall_distance، arc_length، edge_speed و delta می‌سازد، حتی اگر مصرف‌کننده فقط h_film بخواهد. اگر pde_scale و residual_diffusivity هر دو فعال باشند، اندازهٔ همان سرعت نیز دو بار حساب می‌شود.

**چه زمانی رخ می‌دهد؟** شرایط سرعت refresh می‌شوند یا فقط بخشی از فیلدها مصرف می‌شود.

**راه حذف یا کاهش و محدودیت آن:** cache ویژگی‌های واحد هندسه و ساخت خروجی‌های درخواستی؛ استفادهٔ مشترک از speed. δ و damping که به سرعت وابسته‌اند باید با سرعت جدید محاسبه شوند. مسیر fused فعلی نسبت به چهار query جدا بهتر است.

## 10. normal المان نمونه‌برداری دور ریخته و بعداً دوباره پیدا می‌شود

**وضعیت:** قطعی در مسیر مربوط. **محل:** pinnlab/sampling/surface.py:40 (legacy PINNlab: `pinnlab/sampling/surface.py:40`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** BoundaryElementSampler خروجی points, normal از sample_elements را می‌گیرد ولی normal را دور می‌اندازد. اگر بعداً Robin یا GeometryBatchSampler normal بخواهد، از روی مختصات nearest edge یا nearest triangle دوباره پیدا می‌شود.

**چه زمانی رخ می‌دهد؟** نمونه‌برداری element و مصرف‌کنندهٔ نیازمند normal.

**راه حذف یا کاهش و محدودیت آن:** normal همان المان در batch حفظ شود. لازم است معنای normal المانی با normal انتخاب‌شدهٔ مصرف‌کننده یکسان باشد، به‌خصوص روی گوشه‌ها.

## 11. آزمون دوبارهٔ inside/outside در NearWallSampler

**وضعیت:** قطعی برای بخشی از ردیف‌ها. **محل:** pinnlab/sampling/surface.py:331 (legacy PINNlab: `pinnlab/sampling/surface.py:331`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** ابتدا contains(p1) بررسی می‌شود؛ سپس pts با انتخاب p1 یا p2 ساخته می‌شود و contains(pts) دوباره روی همهٔ ردیف‌ها اجرا می‌شود. ردیف‌هایی که p1 در آن‌ها پذیرفته شده قبلاً آزموده شده‌اند.

**چه زمانی رخ می‌دهد؟** NearWallSampler، به‌خصوص با polygon یا mesh بزرگ.

**راه حذف یا کاهش و محدودیت آن:** آزمون دوم فقط برای ردیف‌های fallback مربوط به p2 اجرا شود. نیاز به normal برای ساخت offset در این sampler ضروری است.

## 12. ساخت و آزمودن candidateهای بیشتر از نیاز باقی‌مانده

**وضعیت:** وابسته به نرخ پذیرش. **محل:** pinnlab/sampling/random.py:127 (legacy PINNlab: `pinnlab/sampling/random.py:127`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** SobolSampler در هر دور rejection از n اصلی استفاده می‌کند، حتی وقتی فقط تعداد کمی نقطه باقی مانده است. samplerهای سطحی نیز حداقل‌های ۲۵۶ یا ضرایب oversampling دارند و در پایان بخش اضافی را دور می‌اندازند.

**چه زمانی رخ می‌دهد؟** هندسه با پذیرش کم، یا درخواست تعداد کم نقاط.

**راه حذف یا کاهش و محدودیت آن:** حجم draw بر اساس تعداد باقی‌مانده و نرخ پذیرش تنظیم شود. مقداری oversampling برای کم‌کردن تعداد kernelها مفید است؛ همهٔ نقاط دورریخته‌شده الزاماً خطای طراحی نیستند.

## 13. وزن یکنواختی که در MSE نرمال‌شده حذف می‌شود

**وضعیت:** وابسته به formulation. **محل:** pinnlab/sampling/random.py:49 (legacy PINNlab: `pinnlab/sampling/random.py:49`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** وزن measure/N برای همهٔ نقاط برابر است. در weighted_mse، صورت و مخرج همین ضریب را حذف می‌کنند و نتیجه برابر mean است؛ با این حال اندازه‌گیری هندسه، ساخت N وزن، ضرب و reduction انجام می‌شود. measure بعضی هندسه‌ها Monte Carlo پرهزینه است، هرچند در همان شیء cache می‌شود.

**چه زمانی رخ می‌دهد؟** فقط StrongResidual یا constraint با وزن یکنواخت، نه انتگرال انرژی یا وزن غیریکنواخت.

**راه حذف یا کاهش و محدودیت آن:** در این ترکیب use_weights=False یا مسیر mean برای وزن ثابت. خودِ sample_uniform هندسهٔ عمومی ممکن است برای تخمین نرخ پذیرش همچنان measure بخواهد؛ خاموش‌کردن weights تمام آن هزینه را همیشه حذف نمی‌کند.

## 14. بازسازی اطلاعات ثابت هندسه

**وضعیت:** قطعی با فراخوانی‌های تکراری. **محل:** pinnlab/geometry/polygon.py:142 (legacy PINNlab: `pinnlab/geometry/polygon.py:142`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** Polygon2D.boundary هر بار PolygonBoundary تازه می‌سازد و طول ضلع، normal و normal رأس‌ها را دوباره محاسبه می‌کند. normal و wall_coordinates روی Polygon2D نیز boundary تازه می‌سازند. bounding_box و measure بعضی polygon/meshها هر بار reduction ثابت را تکرار می‌کنند. boundary تازهٔ CSG یا SDF نیز cache اندازهٔ شیء قبلی را به ارث نمی‌برد.

**چه زمانی رخ می‌دهد؟** هندسه تغییر نمی‌کند و این APIها مکرر فراخوانی می‌شوند.

**راه حذف یا کاهش و محدودیت آن:** boundary، bbox، طول‌ها، مساحت و ثابت‌های ray/edge در همان هندسه نگهداری شوند. هندسهٔ قابل‌تغییر نیازمند نسخه و invalidate است. TriangleMeshGeometry._parity نیز ثابت‌های edge/ray را در هر chunk بازمی‌سازد.

## 15. فاصلهٔ unsigned مش، تست علامت هم انجام می‌دهد

**وضعیت:** قطعی در API distance. **محل:** pinnlab/geometry/mesh.py:131 (legacy PINNlab: `pinnlab/geometry/mesh.py:131`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** TriangleMeshGeometry متد unsigned_distance دارد، ولی distance را override نکرده است. متد ارث‌رسیدهٔ Geometry.distance برابر abs(sdf) است؛ sdf علاوه بر فاصله، ray casting داخل/خارج را انجام می‌دهد. abs در پایان علامت را حذف می‌کند.

**چه زمانی رخ می‌دهد؟** wall_distance روی TriangleMeshGeometry.

**راه حذف یا کاهش و محدودیت آن:** override کردن distance با unsigned_distance، همراه با کنترل تطابق معنای فاصله. ray casting برای sdf امضادار همچنان لازم است.

## 16. CSG عضویت همهٔ ردیف‌ها را در همهٔ اعضا می‌سنجد

**وضعیت:** فرصت وابسته به هندسه. **محل:** pinnlab/geometry/csg.py:106 (legacy PINNlab: `pinnlab/geometry/csg.py:106`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** Union، Intersection، Difference و فیلترهای CSGBoundary، contains اعضای بعدی را روی کل نقاط اجرا می‌کنند. در Union، نقطهٔ قبلاً داخل‌شده نیازی به آزمون عضو بعدی ندارد؛ در Intersection، نقطهٔ قبلاً ردشده همین‌طور.

**چه زمانی رخ می‌دهد؟** اعضای زیاد، به‌خصوص اعضای با contains پرهزینه.

**راه حذف یا کاهش و محدودیت آن:** بررسی اعضای بعدی فقط روی ردیف‌های حل‌نشده. indexing و kernelهای کوچک هزینه دارند؛ برای Boxهای ارزان باید با profiling مقایسه شود. برای SDF min/max، حذف اعضا به استدلال جداگانه نیاز دارد.

## 17. Newton پس از رسیدن به مرز هم تعداد ثابت مرحله دارد

**وضعیت:** وابسته به SDF. **محل:** pinnlab/geometry/base.py:133 (legacy PINNlab: `pinnlab/geometry/base.py:133`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** project_to_boundary همهٔ steps را اجرا می‌کند. SDF دقیق می‌تواند در یک مرحله به مرز برسد؛ مرحله‌های بعدی دوباره SDF و گرادیان می‌گیرند. SdfBoundary بعد از projection نیز SDF را برای کنترل پذیرش دوباره می‌گیرد.

**چه زمانی رخ می‌دهد؟** SDF دقیق یا نقاطی که زود همگرا می‌شوند.

**راه حذف یا کاهش و محدودیت آن:** توقف یا active mask با tolerance مناسب؛ برای GPU بررسی همگرایی در هر مرحله می‌تواند خودش sync هزینه‌دار باشد. SDF تقریبی به چند مرحله نیاز واقعی دارد.

## 18. feature و mask می‌توانند query هندسی مشترک را تکرار کنند

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/solvers/features.py:125 (legacy PINNlab: `pinnlab/solvers/features.py:125`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** WallCoordinateFeature نزدیک‌ترین ضلع و فاصله را حساب می‌کند؛ BoundaryLayerWakeMask روی همان نقاط دوباره SDF می‌گیرد. چند field solver با feature/mask هندسی مشترک نیز آن را در هر forward تکرار می‌کنند. GeometryDispatchFeature و GeometryDispatchMask هر کدام unique، ماسک ردیف و subset fields را جدا می‌سازند. AxisFourierFeature هم tensor فرکانس ثابت را هر بار می‌سازد.

**چه زمانی رخ می‌دهد؟** feature و mask یا چند شبکه بر یک هندسه و همان ورودی کار می‌کنند.

**راه حذف یا کاهش و محدودیت آن:** اشتراک intermediate هندسی در همان forward/graph و cache ثابت‌های فرکانس. جایگزین‌کردن feature یا mask مشتق‌پذیر با فاصلهٔ detachشده، مشتق PDE را تغییر می‌دهد؛ cache بین graphها نباید اتصال به points را قطع کند.

## 19. encoder تمام هندسه‌ها را encode می‌کند و cache inference محدود است

**وضعیت:** وابسته به ترکیب. **محل:** pinnlab/solvers/conditional.py:149 (legacy PINNlab: `pinnlab/solvers/conditional.py:149`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** ConditionalFieldSolver جدول latent تمام ensemble را می‌سازد و سپس latent_table[gid] را انتخاب می‌کند؛ اگر فقط یک هندسه فعال باشد، بقیهٔ شکل‌ها هم encode می‌شوند. cache encoder فقط با evaluation token معتبر است. Problem.evaluate و بعضی adapterها این scope را باز نمی‌کنند، بنابراین inference مکرر یا chunkهای UI دوباره encode می‌کنند. DataSourceMSEMonitor scope مشترک برای chunkهای خودش دارد.

**چه زمانی رخ می‌دهد؟** ensemble بزرگ، member فعال کم یا inference chunkشده.

**راه حذف یا کاهش و محدودیت آن:** encode اعضای فعال و lookup محلی؛ در inference cache با نسخهٔ وزن‌ها و ورودی هندسه، یا scope مشترک برای یک عملیات. در آموزش cache گراف پس از backward باید آزاد شود.

## 20. ضرب ماتریس dense برای گراف حلقه‌ای sparse

**وضعیت:** فرصت ساختاری. **محل:** pinnlab/solvers/encoders.py:46 (legacy PINNlab: `pinnlab/solvers/encoders.py:46`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** adjacency گراف فقط رأس خودش و همسایهٔ قبلی و بعدی را دارد، اما A به‌صورت dense ذخیره شده و einsum ضرب کامل را انجام می‌دهد. بیشتر ضرایب صفرند؛ برای V رأس، کار به‌صورت dense با V² رشد می‌کند، در حالی که توپولوژی حلقه‌ای O(V) یال دارد.

**چه زمانی رخ می‌دهد؟** polygon دارای رأس‌های زیاد.

**راه حذف یا کاهش و محدودیت آن:** جمع وزن‌دار همسایه‌ها با roll/index یا sparse matmul. برای V بسیار کوچک، dense ممکن است سریع‌تر باشد. موارد V کوچک و نرمال‌سازی adjacency باید در جایگزین رعایت شوند.

## 21. بهینه‌سازی FiLM همیشه فعال نمی‌شود

**وضعیت:** قطعی در مسیر معمول بدون provider. **محل:** pinnlab/solvers/conditional.py:159 (legacy PINNlab: `pinnlab/solvers/conditional.py:159`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** forward_grouped تنها وقتی استفاده می‌شود که condition_group_field موجود باشد. ConditionalSampler معمولی، حتی با چند scenario تکرارشده، این field را فقط در مسیر provider می‌سازد. در نتیجه FiLM head شرایط یکسان را برای تمام نقاط تکرار می‌کند. validation/airfoil._field_comparison نیز group_id یکتا برای هر ردیف می‌دهد، حتی اگر شرایط فیزیکی برابر باشند.

**چه زمانی رخ می‌دهد؟** FiLMNetwork با ConditionalSampler معمولی یا validation مذکور.

**راه حذف یا کاهش و محدودیت آن:** شناسهٔ گروه صحیح بر اساس هندسه و همهٔ شرایط مؤثر ساخته شود. گروه‌بندی صرفاً با geometry_id می‌تواند شرایط متفاوت را اشتباهاً ادغام کند.

## 22. GAPI grouping و branch برای شرایط یکسان دوباره اجرا می‌شوند

**وضعیت:** وابسته به فراخوانی‌ها. **محل:** pinnlab/solvers/gapi.py:301 (legacy PINNlab: `pinnlab/solvers/gapi.py:301`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** در هر forward، condition scaling، torch.unique روی کلیدها، انتخاب representative و branch_forward تکرار می‌شود. چند batch داخلی/مرزی یا چند chunk inference می‌توانند همان هندسه و شرایط را داشته باشند. داخل حلقهٔ groupها نیز inverse==group_index کل N ردیف را برای هر group پیمایش می‌کند.

**چه زمانی رخ می‌دهد؟** شرایط مشترک بین batchها یا تعداد گروه زیاد.

**راه حذف یا کاهش و محدودیت آن:** اشتراک branch در scope یک graph/نسخهٔ مدل؛ ساخت index گروه‌ها یک بار. trunk نقاط متفاوت را نباید از cache هندسه/شرایط گرفت. GAPI اکنون branch را در یک forward به‌درستی فقط برای گروه‌ها اجرا می‌کند.

## 23. اشتراک AutoGrad با برابری عددی نقاط برقرار نمی‌شود

**وضعیت:** وابسته به assembly. **محل:** pinnlab/problem/problem.py:225 (legacy PINNlab: `pinnlab/problem/problem.py:225`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** ProblemBuilder تنها termهای دارای همان شیء sampler و همان شیء solver را، آن هم با backendهای تماماً AutoGrad، مشترک می‌کند. samplerهای مستقل با seed یکسان، یا selective viewهای تازه برای component یکسان، در این گروه قرار نمی‌گیرند. Domain و ConstraintTerm هر کدام selective view تازه می‌سازند. AutoGrad نیز یک cache جاری دارد، نه cache چند batch.

**چه زمانی رخ می‌دهد؟** چند term واقعاً روی یک تابع و یک batch یکسان کار می‌کنند.

**راه حذف یا کاهش و محدودیت آن:** canonical کردن viewهای انتخاب component و reuse sampler؛ cache مناسب در scope graph. تساوی points کافی نیست: conditions، fields، انتخاب خروجی و نسخهٔ وزن‌ها باید یکسان باشند.

## 24. backend غیر AutoGrad خروجی کامل را برای انتخاب‌های مختلف دوباره حساب می‌کند

**وضعیت:** وابسته به درخواست component. **محل:** pinnlab/derivatives/manager.py:123 (legacy PINNlab: `pinnlab/derivatives/manager.py:123`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** کلید manager شامل (multi_index, components) است. backend غیر AutoGrad ابتدا مشتق همهٔ خروجی‌ها را می‌گیرد و بعد slice می‌کند. دو انتخاب component روی همان multi-index می‌توانند دو محاسبهٔ کامل بسازند.

**چه زمانی رخ می‌دهد؟** FDM، Spectral یا InternalDerivative با چند انتخاب component در یک manager.

**راه حذف یا کاهش و محدودیت آن:** مشتق کامل backend یک بار با کلید مستقل از selection نگهداری و viewها از آن ساخته شوند؛ یا backend انتخاب component را پشتیبانی کند.

## 25. FDM گره‌های مشترک stencil را دوباره ارزیابی می‌کند

**وضعیت:** قطعی با مشتق‌های همپوشان. **محل:** pinnlab/derivatives/fdm.py:84 (legacy PINNlab: `pinnlab/derivatives/fdm.py:84`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** هر derivative خودش solver.forward را برای گره‌های stencil اجرا می‌کند. u_x و u_xx در stencil مرتبهٔ دو هر دو u(x+h) و u(x-h) را می‌خواهند؛ u_xx و u_yy هر دو u(x) می‌خواهند، که D.value هم آن را دارد. cache مشتق نهایی، این اشتراک گره‌ها را ایجاد نمی‌کند.

**چه زمانی رخ می‌دهد؟** درخواست value، dx، dxx و dyy در یک batch.

**راه حذف یا کاهش و محدودیت آن:** cache ارزیابی گره با shift، batch و نسخهٔ وزن‌ها در graph جاری. آزمایش: ۹ forward برای ۵ مجموعه‌نقطهٔ متمایز. FDM ذاتاً به نقاط perturbed نیاز دارد؛ هدف حذف گره‌های تکراری است.

## 26. Spectral برای هر مشتق دوباره forward و FFT می‌گیرد

**وضعیت:** قطعی با چند مشتق. **محل:** pinnlab/derivatives/spectral.py:55 (legacy PINNlab: `pinnlab/derivatives/spectral.py:55`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** هر derivative از solver.forward(batch) آغاز می‌شود و FFT مربوط را محاسبه می‌کند. D.value نیز forward جدا دارد. مشتق‌های اول و دوم یک محور از همان طیف u قابل محاسبه‌اند.

**چه زمانی رخ می‌دهد؟** PDE روی grid دوره‌ای با چند multi-index.

**راه حذف یا کاهش و محدودیت آن:** cache value و طیف محورها یا یک نمایش Fourier مناسب برای درخواست‌ها؛ multiplier هر مشتق جدا اعمال شود. آزمایش value، dx و dy: سه forward یکسان.

## 27. InternalDerivative در DeepONet و FNO شبکه را برای هر مشتق تکرار می‌کند

**وضعیت:** قطعی با چند مشتق. **محل:** pinnlab/solvers/operator.py:97 (legacy PINNlab: `pinnlab/solvers/operator.py:97`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** DeepONet.compute_derivative برای هر multi-index forward و graph تازه می‌سازد؛ FNO.compute_derivative برای هر درخواست دوباره کل _net را اجرا می‌کند. value هم مسیر جدا دارد. اگر input function به‌صورت N ردیف یکسان داده شود، branch معمولی DeepONet آن را N بار پردازش می‌کند؛ ورودی یک‌بعدی مسیر broadcast دارد.

**چه زمانی رخ می‌دهد؟** استفاده از diff:internal با چند مشتق، یا تابع ورودی تکراری در branch.

**راه حذف یا کاهش و محدودیت آن:** اشتراک خروجی FNO و طیف آن؛ graph و گرادیان‌های پایین‌تر مشترک برای DeepONet؛ نگهداری branch input در اندازهٔ گروه. AutoGrad فعلی برای DeepONet pointwise می‌تواند بعضی اشتراک‌ها را فراهم کند.

## 28. PDEهای چندخروجی مشتق‌های مصرف‌نشده می‌گیرند

**وضعیت:** قطعی در معادلات نام‌برده. **محل:** pinnlab/equations/flow.py:111 (legacy PINNlab: `pinnlab/equations/flow.py:111`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** Navier–Stokes dxx و dyy همهٔ [u,v,p] را می‌گیرد، ولی مشتق دوم p را استفاده نمی‌کند. RANS-SA مشتق دوم فشار و mixed derivative برخی خروجی‌ها را مصرف نمی‌کند. Energy dxx/dyy همهٔ [u,v,q,T] را می‌گیرد ولی فقط T را استفاده می‌کند؛ گرادیان سرعت نیز وقتی heating خاموش است مصرف نمی‌شود. انتخاب compact state در component_system هنوز بیش از وابستگی واقعی هر مشتق است.

**چه زمانی رخ می‌دهد؟** این معادلات با backendهایی که برای هر خروجی کار جدا دارند، به‌خصوص AutoGrad.

**راه حذف یا کاهش و محدودیت آن:** dependency هر multi-index به component دقیق اعلام شود. آزمایش scalar sweeps: NS برابر ۹، در حالی که ۷ کافی است؛ Energy بدون heating برابر ۱۲، در حالی که برای q و T و مشتق دوم T مجموعاً ۴ کافی است. این شمارش هزینهٔ forward را حذف‌شدنی فرض نمی‌کند.

## 29. constraintها پیش از slice، همهٔ خروجی‌ها را می‌گیرند

**وضعیت:** وابسته به عرض solver. **محل:** pinnlab/constraints/library.py:239 (legacy PINNlab: `pinnlab/constraints/library.py:239`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** Neumann، Robin و Radiation با component مشخص همچنان D.compute(mi) کامل می‌خواهند و بعد slice می‌کنند. InterfaceValue/InterfaceFlux/PairedRobin تنها component_a/b دارند، ولی components اعلام نمی‌کنند؛ builder در نتیجه selective view لازم را نمی‌سازد. SurfaceForce با viscous=True فقط گرادیان u,v می‌خواهد، اما گرادیان همهٔ خروجی‌ها را درخواست می‌کند.

**چه زمانی رخ می‌دهد؟** solver چندخروجی یا MultiFieldSolver دارای چند شبکه.

**راه حذف یا کاهش و محدودیت آن:** D.value/compute با components دقیق؛ اعلام dependency constraint برای builder. برای شبکهٔ واحد چندخروجی، forward معمولاً همچنان مشترک کامل است؛ صرفه‌جویی اصلی reverse sweep یا شبکه‌های مستقل غیرمرتبط است.

## 30. ParametricBoundary در batch تماماً Dirichlet هم مشتق می‌گیرد

**وضعیت:** قطعی با families چندگانه. **محل:** pinnlab/constraints/library.py:496 (legacy PINNlab: `pinnlab/constraints/library.py:496`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** شرط نیاز به dudn بر اساس وجود نام neumann/robin در پیکربندی است، نه فعال‌بودن آن‌ها در batch. حتی اگر mask همهٔ ردیف‌ها Dirichlet باشد، normal و D.compute اجرا می‌شوند. در batch مختلط، مشتق روی ردیف‌های Dirichlet هم حساب می‌شود.

**چه زمانی رخ می‌دهد؟** families چندگانه با batch تماماً Dirichlet یا مخلوط.

**راه حذف یا کاهش و محدودیت آن:** برای تمام-Dirichlet حذف مشتق؛ برای mixed batch، روش انتخاب ردیف بدون خراب‌کردن cache/graph طراحی شود. families=['dirichlet'] همین حالا مسیر بدون مشتق دارد.

## 31. ضریب callable مشترک چند بار resolve می‌شود

**وضعیت:** قطعی برای callable مشترک. **محل:** pinnlab/constraints/library.py:315 (legacy PINNlab: `pinnlab/constraints/library.py:315`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** Robin به‌صورت پیش‌فرض scale=h دارد، ولی resolve_field(self.h) و resolve_field(self.scale) دو بار callable را اجرا می‌کنند. PairedRobin نیز همین الگو را دارد. resolve_field خود cache ندارد؛ یک forcing/material callable مشترک بین termها یا قسمت‌های residual هم تکرار می‌شود. تکرار field:name فقط lookup است و با callable پرهزینه یکسان نیست.

**چه زمانی رخ می‌دهد؟** h و scale یک callable واحدند، از جمله مثال فعلی all_registries با local_film.

**راه حذف یا کاهش و محدودیت آن:** نتیجهٔ h برای scale همان شیء reuse شود؛ برای تابع ثابتِ وابسته به نقاط، sampler field یا cache scoped مناسب. در صورت callback دارای اثر جانبی، تغییر تعداد call نیاز به قرارداد صریح دارد.

## 32. SA خروجی‌های اضافی و algebra مشترک را دوباره حساب می‌کند

**وضعیت:** وابسته به مصرف‌کننده. **محل:** pinnlab/physics/turbulence.py:94 (legacy PINNlab: `pinnlab/physics/turbulence.py:94`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** eddy_viscosity همیشه derivative تحلیلی dnu_t/dnu_tilde را می‌سازد، حتی اگر caller فقط nu_t را بگیرد. در RANS و Energy روی همان state/conditions، nu، chi، fv1 و derivative دوباره در دو residual محاسبه می‌شوند؛ cache مشتق شبکه شامل این algebra فیزیکی نیست. functions نیز بخشی از همین اطلاعات را بازمی‌سازد.

**چه زمانی رخ می‌دهد؟** کاربرد صرفاً ضریب ویسکوزیته، یا معادلات مشترک RANS و Energy.

**راه حذف یا کاهش و محدودیت آن:** API محاسبهٔ فقط nu_t و intermediate مشترک در graph جاری. در energy متغیر، derivative برای alpha_x/y لازم است و نباید بی‌قید حذف شود. نام‌گذاری خروجی با *_ محاسبه را lazy نمی‌کند.

## 33. وزن صفر یا ضریب صفر لزوماً محاسبات را خاموش نمی‌کند

**وضعیت:** قطعی در مسیرهای نام‌برده. **محل:** pinnlab/problem/problem.py:52 (legacy PINNlab: `pinnlab/problem/problem.py:52`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** compute_losses پیش از ترکیب وزن‌ها term فعال را محاسبه می‌کند؛ FixedWeight(weight=0) آن را خاموش نمی‌کند. Heat با diffusivity=0 همچنان Laplacian می‌گیرد؛ Helmholtz با k=0 value را می‌خواهد. ALAV با c_max=0 یا c_res=0 correction/derivative را می‌سازد. extra_loss مربوط به melt با weight=0 هنوز شبکه و مشتق زمانی را اجرا می‌کند. Poisson.energy با forcing=0 نیز جملهٔ صفرِ forcing*u را می‌سازد؛ با بعضی backendها value اضافه است.

**چه زمانی رخ می‌دهد؟** پیکربندی دارای وزن/ضریب دقیقاً صفر یا شرط سختی که همان شرط مقدار را دقیق enforce کرده و loss متناظر هنوز فعال است.

**راه حذف یا کاهش و محدودیت آن:** خاموش‌کردن term با active/disabled_terms و shortcut ضریب literal صفر با معنای روشن. برای وزن صفر، حذف می‌تواند logging یا strategy را تغییر دهد؛ برای callback دینامیک صرفاً فرض صفر مجاز نیست. ALAV eps.detach نیز graph algebra را بعد از ساخته‌شدن قطع می‌کند؛ محاسبهٔ ضریب frozen می‌تواند از ابتدا روی tensorهای detachشده انجام شود.

## 34. RAR دامنهٔ غیرفعال را هم refine می‌کند و graph آموزش می‌سازد

**وضعیت:** قطعی با دامنهٔ adaptive غیرفعال. **محل:** pinnlab/training/trainer.py:248 (legacy PINNlab: `pinnlab/training/trainer.py:248`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** _refine_adaptive روی همهٔ domains می‌چرخد، بدون کنترل active یا disabled_terms. پس دامنهٔ خارج از phase هم candidate pool و residual می‌گیرد. residual صرفاً برای امتیاز انتخاب نقاط استفاده می‌شود و سپس detach می‌شود، ولی backend AutoGrad همان create_graph=True لازم برای آموزش را استفاده می‌کند.

**چه زمانی رخ می‌دهد؟** RAR همراه phase/disabled terms، یا pool بزرگ.

**راه حذف یا کاهش و محدودیت آن:** refinement با نیاز واقعی phase هماهنگ شود. برای scoring باید مشتق مکانی لازم باقی بماند، اما graph برای backward پارامترهای شبکه تا حد ممکن ساخته/نگهداری نشود؛ torch.no_grad سراسری راه‌حل درستی برای spatial derivatives نیست.

## 35. LBFGS برای گزارش نهایی backward اضافه دارد و termهای خاموش را می‌خواند

**وضعیت:** قطعی در این مسیر. **محل:** pinnlab/training/trainer.py:650 (legacy PINNlab: `pinnlab/training/trainer.py:650`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** closure همیشه compute_losses(0) را بدون active/disabled_terms صدا می‌زند. بنابراین termهای خاموش یا دارای وزن صفر نیز محاسبه می‌شوند. پس از optimizer.step، closure یک بار دیگر فقط برای final loss فراخوانی می‌شود و total.backward را هم انجام می‌دهد، در حالی که دیگر optimizer update وجود ندارد.

**چه زمانی رخ می‌دهد؟** lbfgs_steps>0، به‌خصوص phase/disabled_terms یا مدل گران.

**راه حذف یا کاهش و محدودیت آن:** ارزیابی نهایی loss از closure دارای backward جدا شود و سیاست termهای فعال صریح باشد. تکرار closure هنگام line search معمولاً لازم است، چون وزن‌های آزمایشی تغییر می‌کنند.

## 36. معیارهای خطا prediction و reference مشترک را تکرار می‌کنند

**وضعیت:** قطعی با چند metric. **محل:** pinnlab/benchmark/metrics.py:47 (legacy PINNlab: `pinnlab/benchmark/metrics.py:47`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** L2Relative، MaxError و MeanAbsoluteError هر کدام problem.evaluate(pts) و ref(pts) را جدا اجرا می‌کنند. RegionRMSE و RegionBias نیز هر کدام _diff می‌گیرند، که reference.solve، mask هندسی و prediction روی کل grid را تکرار می‌کند. برای region کوچک، prediction خارج region بعداً دور ریخته می‌شود.

**چه زمانی رخ می‌دهد؟** چند metric روی مدل، نقاط، شرایط و reference یکسان.

**راه حذف یا کاهش و محدودیت آن:** context ارزیابی شامل prediction/reference/difference مشترک باشد؛ reference solver با کلید شرایط cache شود. اگر metrics نقاط یا شرایط متفاوت دارند، ادغام مجاز نیست. آزمایش سه metric: prediction=۳ و reference=۳ فراخوانی.

## 37. BenchmarkRegistry برای context همه‌چیز را دوباره می‌سازد

**وضعیت:** قطعی در run. **محل:** pinnlab/benchmark/registry.py:1436 (legacy PINNlab: `pinnlab/benchmark/registry.py:1436`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** _make_context هندسه، regions و physics را دوباره می‌سازد، هرچند build_problem قبلاً آن‌ها را ساخته است. physics شامل PanelFlow می‌تواند linear solve جدید داشته باشد. همچنین eval_points با n_eval پیش‌فرض ۱۰هزار همیشه نمونه‌برداری می‌شود، حتی اگر تنها metric final_loss یا wallclock باشد.

**چه زمانی رخ می‌دهد؟** BenchmarkRegistry.run، حتی با معیارهای بدون نیاز به نقطه یا reference.

**راه حذف یا کاهش و محدودیت آن:** reuse اجزای assembly و ساخت lazy بر اساس نیاز metric. ساخت reference solver هم فقط وقتی مصرف‌کننده دارد انجام شود. این نکته مربوط به runner استاندارد است؛ مثال سفارشی all_registries مسیر خودش را دارد.

## 38. مرجع FVM و data generation اطلاعات ثابت را در هر condition بازمی‌سازند

**وضعیت:** قطعی با چند solve. **محل:** pinnlab/reference/conjugate_fvm.py:101 (legacy PINNlab: `pinnlab/reference/conjugate_fvm.py:101`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** ConjugateFVMReference.solve هر بار grid، inside/passage، فاصله، arc و اطلاعات رسانایی ثابت را می‌سازد. چهار query جدأ panel نزدیک‌ترین ضلع و کار هندسی را تکرار می‌کنند، در حالی که flow_features مسیر fused دارد. در u_inf ثابت و t_inf متفاوت، ماتریس A یکسان است ولی assembly و sparse factorization تکرار می‌شود. ConjugateFVMDataSource بعد از solve همان wall_distance grid را دوباره برای poolها می‌گیرد و در condition بعد هم تکرار می‌کند. SU2 solve و field نیز هر دو _run مستقل دارند و اگر هر دو برای یک case خواسته شوند کل CFD دو بار اجرا می‌شود.

**چه زمانی رخ می‌دهد؟** چند condition روی هندسه ثابت، چند metric مرجع یا درخواست همزمان نیرو و میدان SU2.

**راه حذف یا کاهش و محدودیت آن:** precompute هندسه، استفادهٔ fused با بررسی سازگاری عددی، cache solve/factorization با کلید شرایط مؤثر و reuse فاصله برای data pools. تغییر u_inf ممکن است A را تغییر دهد. flow_features با مسیرهای جدا در guardهای فاصله تفاوت‌های ریز دارد؛ جایگزینی نیازمند parity check است. منابع دوم: data/conjugate.py:209 و reference/su2_solver.py:247.

## 39. بعضی حلقه‌های مرجع بعد از همگرایی هم ادامه می‌دهند

**وضعیت:** قطعی برای tol نام‌برده. **محل:** pinnlab/reference/fvm_core.py:49 (legacy PINNlab: `pinnlab/reference/fvm_core.py:49`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** _cg_poisson پارامتر tol را می‌گیرد ولی آن را در حلقه استفاده نمی‌کند؛ حتی rhs=0 همهٔ max_iterها اجرا می‌شوند. solve_scalar_laplace در airfoil_fvm نیز tol را مصرف نمی‌کند. CG داخلی solve_airfoil_flow تعداد ثابت cg_iter دارد. در scalar Laplace، aP و چند مقدار ثابت در هر iteration بازساخته می‌شوند؛ در fvm_core.momentum چهار roll ابتدایی پیش از مصرف overwrite می‌شوند و coefficientهای ثابت هر call دوباره ساخته می‌شوند.

**چه زمانی رخ می‌دهد؟** solveهای خیلی آسان، همگراشده یا تکرارشونده.

**راه حذف یا کاهش و محدودیت آن:** توقف با residual یا چک دوره‌ای و precompute ثابت‌ها؛ هزینهٔ synchronization GPU با هزینهٔ iteration مقایسه شود. outer SIMPLE و solve_energy هم‌اکنون چک دوره‌ای همگرایی دارند؛ همهٔ تکرارهای numerical solver اضافی نیستند.

## 40. monitor و DataFit ردیفی، خروجی‌های دورریختنی را هم می‌گیرند

**وضعیت:** وابسته به solver. **محل:** pinnlab/training/monitors.py:84 (legacy PINNlab: `pinnlab/training/monitors.py:84`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** monitor ابتدا solver.forward همهٔ خروجی‌ها را می‌گیرد و سپس output_component را gather می‌کند. DataFit با component به‌صورت field هم همین کار را می‌کند. برای MultiFieldSolver، یک ردیف solid می‌تواند شبکهٔ fluid را نیز اجرا کند و برعکس. چند monitor روی همان source هر کدام دوباره prediction می‌گیرند.

**چه زمانی رخ می‌دهد؟** منبع مخلوط ردیف‌های component، چند sub-network مستقل یا چند monitor هم‌پوشان.

**راه حذف یا کاهش و محدودیت آن:** dispatch ردیف‌ها به sub-solver لازم و اشتراک prediction monitorهای یکسان. splitهای داخل یک DataSourceMSEMonitor همین حالا از یک squared-error مشترک استفاده می‌کنند و forward اضافه ندارند. monitor پس از optimizer update است؛ خروجی قبل از update آموزش جایگزین صحیح آن نیست.

## 41. mask ناحیه برای سه panel یک شکل سه بار حساب می‌شود

**وضعیت:** قطعی با region_contours. **محل:** pinnlab/visualization/fields.py:93 (legacy PINNlab: `pinnlab/visualization/fields.py:93`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** حلقهٔ سه subplot، برای هر geometry موجود در region_contours دوباره geometry.contains(points) را اجرا می‌کند. reference، prediction و difference روی همان grid قرار دارند؛ mask هر سه یکسان است.

**چه زمانی رخ می‌دهد؟** plot_reference_comparison با region contour.

**راه حذف یا کاهش و محدودیت آن:** mask هر ناحیه پیش از حلقهٔ panelها یک بار محاسبه شود. خودِ رسم contour روی هر محور لازم است، query هندسهٔ آن لازم نیست تکرار شود.

## 42. UI سه interpolation روی همان مختصات انجام می‌دهد

**وضعیت:** قطعی در comparison. **محل:** src/turbine_generalized_ui.py:721 (legacy PINNlab: `src/turbine_generalized_ui.py:721`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** برای theta، pinn_theta و absolute_error سه griddata جدا با همان source coordinates و grid points اجرا می‌شود؛ triangulation خطی و ساخت nearest index می‌توانند مشترک باشند. هر component نیز برای کل grid interpolation می‌گیرد و بعد فقط metal/gas را نگه می‌دارد.

**چه زمانی رخ می‌دهد؟** صفحهٔ comparison توربین با دادهٔ پراکنده.

**راه حذف یا کاهش و محدودیت آن:** یک triangulation و نگاشت مشترک برای چند ستون، interpolation فقط روی region لازم و nearest فقط برای queryهای missing. interpolation(|error|) با |interpolation(pred)-interpolation(gt)| برابر نیست؛ ستون خطا نباید با این تبدیل عوض شود.

## 43. UI و inference خروجی‌های نامعتبر یا دوباره‌خواسته‌شده را محاسبه می‌کنند

**وضعیت:** وابسته به استفاده. **محل:** src/conditional_heatmap_ui.py:1241 (legacy PINNlab: `src/conditional_heatmap_ui.py:1241`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** heatmap ابتدا کل bounding-box grid و همهٔ fieldها را evaluate می‌کند و بعد valid/region mask می‌زند. MultiFieldAdapter همهٔ sub-solverها را روی همهٔ نقاط اجرا می‌کند، حتی برای نمایش یک field یا ردیف‌هایی که بعداً پوشانده می‌شوند. _predict توربین هم solid و fluid را روی همهٔ نقاط می‌گیرد و سپس یکی را انتخاب می‌کند. مدل و جدول GT cache دارند، ولی prediction/interpolation عمدتاً با rerun UI تکرار می‌شوند، حتی اگر ورودی فیزیکی تغییر نکرده باشد؛ force readout هم می‌تواند با تغییر صرفاً resolution دوباره محاسبه شود.

**چه زمانی رخ می‌دهد؟** هندسهٔ باریک، grid بزرگ، انتخاب یک field یا interaction نمایشی در Streamlit.

**راه حذف یا کاهش و محدودیت آن:** انتخاب field و valid rows پیش از inference، cache نتیجه با کلید model version، هندسه، شرایط، زمان و نقاط؛ force cache مستقل از resolution. prediction روی grid و نقاط GT متفاوت را نباید یک محاسبه فرض کرد.

## 44. تبدیل تکراری همان loss به عدد برای logging

**وضعیت:** کم‌اثر و وابسته به دستگاه. **محل:** pinnlab/training/trainer.py:493 (legacy PINNlab: `pinnlab/training/trainer.py:493`).

**چه محاسبه‌ای تکرار یا اضافه می‌شود؟** در step ثبت، _log، metrics و bar.set_postfix همان lossهای tensor را جدا به float تبدیل می‌کنند. روی CUDA این readbackها و چک‌های Python روی tensor می‌توانند synchronization داشته باشند. این تکرار شبکه نیست، ولی به‌ویژه برای مدل‌های کوچک هزینهٔ اجرایی دارد؛ در حالت progress=False هم دادهٔ postfix ساخته می‌شود.

**چه زمانی رخ می‌دهد؟** logging پرتکرار، CUDA و شبکهٔ کوچک.

**راه حذف یا کاهش و محدودیت آن:** lossهای scalar یک بار به host منتقل و برای history/logger/progress استفاده شوند؛ postfix وقتی progress خاموش است ساخته نشود. چک finite برای robustness ارزش دارد و حذف بی‌قید آن توصیهٔ گزارش نیست.

## تکرارهایی که لازم‌اند یا اکنون کنترل می‌شوند

- تغییر وزن‌های شبکه پس از optimizer.step، مقدار و مشتق قدیمی را نامعتبر می‌کند؛ ثابت‌بودن points اجازهٔ reuse نتیجهٔ شبکه در همهٔ stepها را نمی‌دهد.
- نقاط interior و boundary متفاوت‌اند و کاربردهای متفاوت دارند؛ forward روی این دو مجموعه صرفاً به‌خاطر شباهت کد تکرار زائد نیست.
- AutoGrad برای یک component و graph، گرادیان کامل مختصات را می‌گیرد و میان درخواست‌های محورهای مختلف reuse می‌کند. گرفتن ux و uy به‌خودی‌خود دو sweep نیست. مشتق‌های مرتبهٔ بالاتر از cache والد استفاده می‌کنند.
- GradNorm و NTK فقط در stepهای refresh تعیین‌شده gradient جدأ termها می‌گیرند. این هزینهٔ روش وزن‌دهی است؛ FixedWeight آن را ندارد.
- closureهای line search با وزن‌های آزمایشی متفاوت LBFGS به ارزیابی تازه نیاز دارند. مورد ۳۵ دربارهٔ فراخوانی اضافی نهایی و termهای خاموش است.
- TrainingDataSource.load cache حافظه دارد؛ ConjugateFVMDataSource cache دیسک و قفل بین processها دارد. فراخوانی load الزاماً حل دوبارهٔ مرجع نیست.
- PhysicsFieldSampler از flow_features fused استفاده می‌کند. مشکل باقی‌مانده refresh همان اطلاعات ثابت یا ساخت خروجی‌های غیرمصرفی است.
- GeometryDispatchFeature و GeometryDispatchMask فعلی فقط هندسه‌ها و ردیف‌های فعال را evaluate می‌کنند؛ توضیح قدیمی بعضی docstringها دربارهٔ محاسبهٔ همهٔ هندسه‌ها، رفتار فعلی کد نیست.
- MultiFieldSolver پارامترهای moduleهای مشترک را deduplicate می‌کند؛ optimizer از این مسیر encoder مشترک را چند بار update نمی‌کند.
- بعضی حل‌گرهای مرجع بررسی همگرایی را دوره‌ای انجام می‌دهند تا GPU هر iteration با CPU sync نشود. ادامهٔ چند iteration بین دو چک می‌تواند انتخاب عملکردی عمدی باشد.

## اولویت پیشنهادی

ابتدا موارد ۱، ۲۵، ۲۶، ۲۷، ۲۸، ۲۹، ۳۶، ۳۷، ۳۸ و ۳۹ بررسی شوند، چون یا حذف روشن دارند یا امکان حذف forward/derivative/solve گران را فراهم می‌کنند. پس از آن cache هندسه و sampler و inference. صرفه‌جویی واقعی باید برای پیکربندی هدف اندازه‌گیری شود؛ تعداد بیشتر موارد کم‌اثر لزوماً زمان بیشتری از یک solve تکراری نمی‌گیرد.
