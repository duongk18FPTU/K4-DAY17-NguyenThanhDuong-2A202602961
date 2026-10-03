# K4-DAY17-NguyenThanhDuong-2A202602961

Bài cá nhân Day 17 — Memory Systems for AI Agent.

- Họ tên: Nguyễn Thanh Dương
- MSSV: 2A202602961
- Khóa: K4
- Phân tích Bước 8, bonus và tự đối chiếu Rubric: [STEP8.md](STEP8.md)

## Cấu trúc bài nộp

```text
K4-DAY17-NguyenThanhDuong-2A202602961/
├── src/                   # Python implementation và test
├── data/                  # Hai dataset gốc, giữ nguyên
├── STEP8.md               # Bốn câu phân tích và ba câu trả lời bonus
├── README.md
├── requirements.txt
└── .gitignore
```

`src/` và `data/` đủ để chạy benchmark và test offline. Các file báo cáo/phân tích
không tham gia tính điểm. `.env` và `state/` không phải dữ liệu bài nộp.

## Setup

Khuyến nghị Python >= 3.11. Offline đã được kiểm chứng trên Python 3.10.8.

```bash
python -m pip install -r requirements.txt
```

Nếu chỉ chép `src/` và `data/`, cài `pytest` bằng `python -m pip install pytest`.
Benchmark offline chỉ dùng standard library; `python-dotenv` là tùy chọn khi không
có `.env`. Không cần API key hoặc provider SDK cho hai lệnh kiểm tra.

## Hai lệnh kiểm tra từ thư mục gốc

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Benchmark mặc định offline, in đúng hai bảng **Standard Benchmark** và
**Long-Context Stress Benchmark**, mỗi bảng hai dòng Baseline/Advanced với sáu metric:
Agent tokens only, Prompt tokens processed, Cross-session recall, Response quality,
Memory growth (bytes), Compactions.

Test đã pass **34/34**, bao gồm bốn test cốt lõi: đọc/ghi/sửa `User.md`, compact
trigger, cross-session recall và giảm prompt load. Hai lệnh trên đã chạy thành công
trên bản sao sạch chỉ có `src/*.py` và `data/*.json`. Không có `.env` hoặc `state/`
ban đầu; state sinh ra đã được xóa trước khi chạy test.

## Kết quả và thiết kế

Advanced đạt 100% recall trên hai dataset; Baseline đạt 0% ở thread mới.
Standard: Advanced 36,981 prompt tokens, Baseline 26,131, chưa có compaction.
Stress: Advanced 14,235 prompt tokens, Baseline 28,592; compact 6 lần, giảm 50.2%.
Threshold 1,000 estimated tokens; giữ 4 message gần nhất. Token offline là heuristic,
quality là điểm recall/brevity, không phải phép đo chất lượng ngữ nghĩa độc lập.

- **Baseline:** lịch sử đầy đủ trong RAM theo thread, không có persistent memory.
- **Advanced:** `User.md` theo user ID sống qua restart; recent messages và summary
  trong RAM theo thread. Summary giữ facts tích lũy và tối đa sáu topic snippets.
- **Bonus:** entity extraction có field, dedup và conflict handling; correction thay
  fact cũ. Gate lọc câu hỏi, giả định, quote, phủ định và câu đùa trước khi ghi.

Hai agent dùng chung extractor, response policy và token estimator. Mỗi recall
question được hỏi ở thread mới riêng. Mỗi agent/suite dùng state tạm sạch; benchmark
không đọc hoặc xóa profile cá nhân có sẵn. Dataset giữ nguyên so với đề bài.

## Phép đo thêm và live extension

```bash
python src/benchmark.py --ablation --output reports/benchmark.md --json reports/benchmark.json
```

Adapter live hỗ trợ openai, custom, gemini, anthropic, ollama, openrouter. Cài package
LangChain của provider được chọn và đặt LLM_PROVIDER, LLM_MODEL, credential qua biến
môi trường. Benchmark chỉ gọi provider khi có `--live`; agent gọi trực tiếp dùng
`LAB_OFFLINE=false`. SDK/credential errors được báo rõ, không âm thầm chuyển offline.
Memory do lab quản lý, LangChain cung cấp model transport. Live được test bằng mock,
chưa gọi API thật; judge config có nhưng quality hiện vẫn dùng heuristic.
