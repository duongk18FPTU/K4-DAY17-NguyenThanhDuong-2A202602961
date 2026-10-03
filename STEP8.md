# Bước 8 — Phân tích kết quả và bonus

Nguyễn Thanh Dương — MSSV 2A202602961 — K4.

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ có lịch sử trong từng thread. Recall được hỏi trong thread mới riêng
cho mỗi câu hỏi nên không có facts từ phiên trước. Advanced đọc `User.md` theo user ID,
vì vậy vẫn nhớ sau khi đổi thread và sau khi tạo lại instance agent.
Trên hai dataset, recall Baseline là 0%, Advanced là 100%.
Hai agent dùng cùng input, extractor và response policy; đáp án chuẩn chỉ được
benchmark sử dụng để chấm điểm, không được đưa cho agent.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Standard có 10 conversation, 101 lượt và 14 recall probes. Các thread chưa vượt
threshold 1,000 tokens nên không có compaction. Advanced vẫn phải đưa profile và
hướng dẫn memory vào mỗi prompt: 36,981 prompt tokens so với 26,131 của Baseline,
tăng khoảng 41.5%. Output Advanced là 1,796 tokens, Baseline là 1,860; output ngắn
hơn không đồng nghĩa tổng input cost thấp hơn.

## 3. Vì sao compact giúp Advanced ở hội thoại dài?

Stress có 16 lượt dài và 3 recall probes. Baseline giữ toàn bộ lịch sử, xử lý lại
ngữ cảnh cũ ở mỗi lượt. Advanced compact 6 lần, thay phần cũ bằng summary tích lũy
và giữ 4 message gần nhất. Prompt tokens giảm từ 28,592 xuống 14,235, khoảng 50.2%,
trong khi recall Advanced vẫn là 100%.

Phép đo thêm khi tắt compact trên cùng Advanced cho 30,278 prompt tokens và 100%
recall. Bật compact giảm khoảng 53.0% prompt load so với cấu hình đó. Output không
giảm tương ứng: Advanced 1,060, Baseline 990, Advanced tắt compact 1,026 tokens.
Lợi ích chính là giảm **Prompt tokens processed**, không phải luôn giảm output.
Nếu thay summarizer cục bộ bằng LLM, cần cộng thêm token của việc tóm tắt.

## 4. File memory tăng trưởng ra sao và có rủi ro gì?

Profile tăng 273 bytes trong Standard và 236 bytes trong Stress. Có tám field cố
định; mỗi field giữ một giá trị hiện tại, tối đa 512 ký tự. Ghi lại cùng giá trị
không làm file tăng; correction thay giá trị cũ. Metric memory growth đo riêng
`User.md`, không đo summary hay RAM. Recent messages và summary chỉ sống trong
thread; các tin tức, mã dự án và deadline tạm thời không được lưu vào profile.

Rủi ro gồm lưu nhầm fact từ câu hỏi/câu đùa, giữ thông tin cũ sai, tăng số profile
và thread theo thời gian, mất chi tiết khi compact. Gate lọc khai báo không chắc,
upsert theo field và giới hạn topic snippets giảm rủi ro nhưng không loại bỏ hết.
Atomic replace tránh file bị ghi một phần; chưa có lock nhiều writer đồng thời.
Production còn cần provenance, retention và quyền xóa memory.

## Bonus — Entity extraction và conflict handling

### Bonus giải quyết vấn đề gì?

Facts được tách thành `name`, `location`, `profession`, `response_style`, `interests`,
`favorite_drink`, `favorite_food`, `pet`. Mỗi field giữ một giá trị mới nhất. Khi
người dùng correction, profile không giữ đồng thời nghề/nơi ở cũ và mới.
Gate lọc câu hỏi (kể cả không có dấu `?`), phủ định, giả định, quote và câu đùa.

### Nó cải thiện recall hoặc token cost thế nào?

Standard cập nhật Đà Nẵng thành Huế và backend engineer thành MLOps engineer.
Stress cập nhật Huế thành Đà Nẵng, bỏ qua Hà Nội là chuyến công tác và product
manager là câu đùa. Đáp án recall không chứa fact cũ sai; các test xác minh điều
này. Upsert và dedup kiểm soát độ dài profile thay vì append mọi message, giúp
giảm nguy cơ profile làm prompt phình lên. Kết quả 100% recall đo được trên dataset
mẫu; chưa có ablation riêng để lượng hóa phần đóng góp của từng bonus.

### Nó tạo thêm rủi ro gì?

Regex có thể bỏ sót cách diễn đạt mới hoặc hiểu sai câu phức tạp. Last-write-wins
chưa phân biệt thời điểm có hiệu lực và độ tin cậy của nguồn. Giới hạn field có
thể bỏ qua thông tin hữu ích. Gate là rule, không phải confidence score được hiệu
chuẩn; chưa có memory decay. Hướng mở rộng là provenance/timestamp và structured
extraction có đánh giá confidence trên dữ liệu độc lập.

## Cách đo và giới hạn

Offline token là heuristic `ceil(chars / 4)` cộng framing overhead chung cho hai
agent, không phải tokenizer/billing thực. Recall chấm expected substrings, có
chuẩn hóa Unicode và bỏ qua hoa/thường. Quality là heuristic recall/brevity, không
phải đánh giá ngữ nghĩa độc lập. Summary là nén mất thông tin; threshold là ngưỡng
trigger mềm. Live provider mới được test bằng mock, chưa gọi API thật.

## Kiểm tra từ trạng thái sạch

Đã chạy đúng hai lệnh dưới đây trên bản sao chỉ chứa các file Python trong `src/`
và hai JSON gốc trong `data/`, không có `.env`, `state/`, `reports/` hoặc file phân
tích. Sau benchmark, đã xóa thư mục state vừa sinh (rỗng) trước khi chạy test.

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Benchmark in đúng hai bảng Standard/Long-Context Stress, mỗi bảng hai dòng
Baseline/Advanced và sáu metric. **34 test pass**, bao gồm cả bốn test cốt lõi của đề.
Môi trường đã kiểm chứng: Windows, Python 3.10.8, pytest 9.1.1.

## Đối chiếu bốn mốc Rubric

| Mốc | Kết luận tự kiểm | Bằng chứng |
| --- | --- | --- |
| 0–60 | Đạt yêu cầu | Baseline, Advanced có User.md, compact, hai dataset và cấu trúc module rõ ràng |
| 60–75 | Đạt yêu cầu | Cùng input; test User.md, trigger, recall; bảng đủ sáu metric |
| 75–90 | Đạt yêu cầu | Hai suite; stress 6 compactions, giảm 50.2% prompt load; phân tích overhead Standard và Prompt tokens processed |
| 90–100 | Có cơ sở vào dải này | Entity extraction và conflict handling; bonus trả lời vấn đề, tác dụng, rủi ro; test correction/noise/dedup |

Đây là tự đối chiếu điều kiện dải 90–100, không phải điểm 100 đã được giảng viên xác nhận.
