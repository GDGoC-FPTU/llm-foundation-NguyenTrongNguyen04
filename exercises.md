# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Mình thấy để temperature = 0.0 thì câu trả lời khá “an toàn”, ít đổi khác và thường đi thẳng vào ý. Càng tăng lên 0.5, 1.0, 1.5 thì câu chữ bắt đầu phóng khoáng hơn, nhiều chi tiết hơn, nhưng đôi khi hơi lan man hoặc thêm mấy ý không thật sự cần.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ để khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng chủ yếu cần trả lời đúng và đều, nên mình muốn nó ổn định hơn là “sáng tạo”; để thấp một chút vẫn giúp câu trả lời không bị quá khô.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Nếu nhìn giá trong `template.py` thì GPT-4o đắt hơn GPT-4o-mini khoảng \(5/0.15 \approx 33.3\) lần ở input và \(20/0.6 \approx 33.3\) lần ở output. Nên với workload này mình ước tính cũng khoảng **33 lần** (xấp xỉ vậy, vì tỉ lệ input/output đều ra gần như nhau).

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> Mình thấy GPT-4o đáng tiền khi câu hỏi khó và “sai là mệt”, ví dụ đọc yêu cầu dài rồi suy luận ra bước làm/điều kiện, hoặc tóm tắt tài liệu mà cần ít lỗi. Còn GPT-4o-mini hợp hơn cho mấy việc nhẹ và nhiều như FAQ, trả lời câu hỏi đơn giản, hoặc làm bản nháp nhanh trước khi cần thì mới chuyển sang GPT-4o.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Theo mình streaming quan trọng nhất khi câu trả lời dài (hoặc người dùng đang chat liên tục) vì nhìn chữ chạy ra sẽ đỡ “đứng hình”, cảm giác nhanh hơn. Còn non-streaming thì hợp khi câu trả lời ngắn, hoặc khi mình cần lấy nguyên cục output để xử lý tiếp (ví dụ yêu cầu trả JSON để parse/validate), hay chạy kiểu batch/background thì streaming không cần thiết lắm.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
