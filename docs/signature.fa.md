# Signature، محورهای مستقل و گروه محورها با ...

**قاعدهٔ استقلال:** حفظ نام یک محور بین ورودی و خروجی هر مدل، تضمین نویسندهٔ مدل دربارهٔ استقلال نمونه‌ها در آن محور است؛ نه فقط برابری تعداد نمونه‌ها. این قاعده هم برای محور core و هم برای محورهای خودکار و wildcard اعمال می‌شود. تابعی که نقاط را با هم ترکیب می‌کند، مانند attention یا کم‌کردن میانگین تمام نقاط، باید نام محور خروجی را عوض کند؛ مثلاً point به query. engine الگوریتم دلخواه کاربر را برای اثبات این ادعا بازرسی نمی‌کند.

## مسیر معمول

```python
Signature({x: ()}, {y: ()})                # Pointwise; automatic mapping.
Signature({x: "point"}, {y: "point"})      # Full point axis; promises independence.
Signature({x: "point"}, {y: ()})           # Reduce the full point axis.
Signature({x: "point"}, {y: "query"})      # May mix points; output has a new axis.
```

evaluate در این مسیر Tensor یا لیست Tensor خام می‌گیرد؛ محورهای خارج از Signature خودکار با vmap یا حلقهٔ لیست پردازش می‌شوند. موتور ترتیب core را می‌چیند. شکل ذاتی Quantity مثل (3,) جزو محورهای نمونه نیست.

## محور ویژهٔ "..."

این یک رشتهٔ ویژه در Signature است، نه آرگومان all_axes=True و نه نام یک محور واقعی Value. حداکثر یک بار در هر ورودی یا خروجی می‌آید. اگر Signature شامل آن باشد، مدل مسئول batching می‌شود: forward همان نگاشت ورودی و همان Valueها را یک بار به evaluate می‌دهد؛ هیچ برش، جابه‌جایی، حلقه یا vmap خودکاری اجرا نمی‌کند. خروجی evaluate نیز باید نگاشت Quantity به Value باشد. فقط payloadهای پشتیبانی‌شدهٔ Value، یعنی Tensor و لیست تخت Tensor، مجازند؛ نوع عمومی جدیدی اضافه نشده است.

```python
class Square(Model):
    signature = Signature({x: "..."}, {y: "..."})

    def evaluate(self, values):
        value = values[x]
        tensor = ([t.square() for t in value.tensor]
                  if isinstance(value.tensor, list) else value.tensor.square())
        return {y: Value(tensor, value.axes)}
```

"..." تمام نام محورهای نمونهٔ ورودی‌های wildcard را که جای دیگری در Signature صریح نیامده‌اند می‌گیرد. محورهای ذاتی را شامل نمی‌شود و ممکن است گروه خالی باشد. در چند ورودی، هرکدام می‌تواند زیرمجموعه‌ای از این محورهای مشترک را داشته باشد؛ خود مدل broadcasting و ترتیب‌ها را مدیریت می‌کند. ترتیب گروه خروجی از اولین ظهور محور در ورودی‌ها تعیین می‌شود. در هر ورودی، ترتیب خود Value حفظ می‌شود.

مثلاً برای x با محورهای (scenario, point):

| ورودی Signature | خروجی Signature | قرارداد |
|---|---|---|
| "..." | "..." | حفظ تمام محورها و استقلال نمونه‌ها |
| "..." | () | تجمیع تمام محورهای نمونه |
| ("...", "point") | "..." | تجمیع point؛ حفظ و استقلال scenario |
| ("...", "point") | ("...", "point") | حفظ هر دو محور و استقلال روی هر دو |
| ("...", "point") | ("...", "query") | پردازش وابستهٔ نقاط؛ حفظ استقلال scenario |

مثال کاهش نقاط با حفظ سناریوها:

```python
class MeanPoints(Model):
    signature = Signature({x: ("...", "point")}, {y: "..."})

    def evaluate(self, values):
        value = values[x]
        # Here point must be the final sample axis as declared above.
        if isinstance(value.tensor, list):
            tensor = [t.mean(dim=-1) for t in value.tensor]
        else:
            tensor = value.tensor.mean(dim=-1)
        return {y: Value(tensor, value.axes[:-1])}
```

این مثال برای Quantity اسکالر است. برای بردارها جای بعد ذاتی هم لحاظ شود. ترتیبِ نام‌های صریح نسبت به "..." باید با ورودی واقعی تطابق داشته باشد؛ چون موتور در این مسیر عمداً Value را دست‌نخورده می‌دهد. برای فقط "..." هر ترتیب واقعی پذیرفته است. ورودی بدون wildcard در همین Signature باید دقیقاً محورهای صریحش را داشته باشد؛ اگر قرار است محور اضافی هم بگیرد، برای آن هم "..." بنویسید. وقتی خروجی "..." دارد، تمام گروه مشترک در آن حفظ می‌شود؛ گروه مستقل دیگری مانند ...1 در این نسخه نداریم.

Shape ذاتی، کلیدهای خروجی، طول محورهای هم‌نام و تطابق محورهای خروجی با Signature همچنان بررسی می‌شوند. خروجی با حذف "..." نباید محورهای گروه را پنهانی نگه دارد. نام یا تعداد نقاط عوض‌شده، نام محور تازه می‌خواهد. هیچ تضمینی برای کامپایل یک evaluate دلخواه وجود ندارد؛ تست fullgraph با backend eager فقط ثبت گراف نمونه را بررسی می‌کند.

Solver نیاز به شاخهٔ اجرای تازه ندارد؛ همچنان فقط model(values) را صدا می‌زند. AD از نام‌های واقعی Valueها استفاده می‌کند: محورهای تغییرنام‌یافته Jacobian کامل می‌خواهند، محورهای حفظ‌شده مشتق متناظر. کاهش و سپس broadcast دوباره، سابقهٔ وابستگی محور را پاک نمی‌کند. قابلیت‌های قبلی مشتق مرتبهٔ بالاتر و گرادیان آموزش حفظ می‌شوند.
