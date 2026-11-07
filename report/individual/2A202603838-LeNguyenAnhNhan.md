# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603838
- **Họ và tên:** Lê Nguyễn Anh Nhân
- **SEED:** 489 (Sinh ra từ MSSV: 2A202603838)

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ Có |

## 2. Kết quả chính

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (lô) | p50 (ms) | p95 (ms) | USD/1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 1.000 | 0.832 | 0.908 | 0.947 | 9.212 | 4913.82 | 0.06 | 0.31 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.990 | 0.935 | 0.962 | 0.974 | 9.314 | 22011.52 | 0.33 | 1.10 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.875 | 0.981 | 0.925 | 1.000 | 3.710 | 209.30 | 13.81 | 17.45 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.981 | 0.972 | 0.977 | 0.974 | 4.078 | 1327.76 | 6.84 | 10.24 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 1.000 | 0.963 | 0.981 | 0.974 | 4.796 | 1502.56 | 6.50 | 7.42 | 0.000 |
| 4a. LLM nhỏ zero-shot (gpt-oss-20b) | 1.000 | 0.950 | 0.974 | 0.974 | 0.000 | 0.81 | 514.73 | 3998.91 | 0.008 |
| 4b. LLM lớn zero-shot (gpt-oss-120b) | 1.000 | 0.950 | 0.974 | 0.974 | 0.000 | 0.74 | 553.28 | 3872.65 | 0.015 |
| 4c. LLM nhỏ few-shot | 0.905 | 0.950 | 0.927 | 0.927 | 0.000 | 0.71 | 556.99 | 3850.96 | 0.015 |

## 3. Dự đoán và thực tế

- **Dự đoán `llm_fewshot` (Đoán: Few-shot tăng nhẹ F1 hoặc giữ nguyên; chi phí phình to):**
  * Thực tế: Điểm F1 của few-shot thực chất bị sụt giảm từ 0.974 xuống 0.927 (so với zero-shot), trong khi chi phí tăng gấp đôi ($0.00031 lên $0.00060) và tốc độ xử lý bị chậm lại.
  * Lý do lệch: Việc đưa 4 ví dụ tĩnh từ tập train không làm mô hình thông minh hơn đối với các câu dạng biên phức tạp trên SUB, ngược lại có thể gây nhiễu ranh giới quyết định của LLM nhỏ (gpt-oss-20b) và tăng chi phí phân tích ngữ cảnh không cần thiết.

- **Dự đoán `hard_eval` (Đoán: BoW đạt khoảng 60% trên 50 case khó):**
  * Thực tế: BoW + LogReg chỉ đạt F1 là 0.333 (accuracy 52%) và BoW + MLP đạt 0.421 (accuracy 56%).
  * Lý do lệch: Hiệu năng thực tế sụt giảm nghiêm trọng hơn dự kiến rất nhiều vì bẫy ngụy trang trong 50 case khó triệt tiêu hoàn toàn thế mạnh từ điển tĩnh của Bag-of-Words (ví dụ tách khoảng trắng, viết leetspeak).

- **Dự đoán `dl_relu` (Đoán: Bỏ ReLU làm hai lớp Linear tương đương một phép affine; F1 sẽ gần LogReg):**
  * Thực tế: Không ReLU đạt F1 = 0.957, vẫn cao hơn rõ rệt so với 0.908 của LogReg trên BoW.
  * Lý do lệch: Dù về mặt toán học 2 lớp Linear không kích hoạt phi tuyến sẽ suy giảm thành một phép biến đổi tuyến tính đơn lẻ, việc tối ưu hóa bằng thuật toán Adam trong 30 epochs và kiến trúc phân rã ma trận trung gian qua 64 chiều (Linear 2000->64->1) hoạt động như một bộ lọc nhiễu ngẫu nhiên, giúp mô hình hội tụ tốt hơn so với LogReg thuần túy chạy gradient descent thông thường.

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** `1. BoW + LogReg` (F1 từ **0.908** thường xuống **0.333** trên tập khó).
- **Nhóm case khó nhất:** Nhóm **Né từ khoá (Obfuscation)** (độ chính xác 0%) và **Tưởng spam nhưng không phải (Ham looks spam)**.

- **Hai case cụ thể lỗi điển hình:**
  1. **Case #2 (Ham looks spam) - "FREE pizza in the office kitchen right now..." (Nhãn thật: ham, BoW đoán: spam):** Do chứa quá nhiều từ khóa đặc trưng có trọng số dương rất lớn từ tập huấn luyện như `FREE`, `now` và mang tính thúc giục. Mô hình BoW không nắm giữ ngữ cảnh đồng nghiệp nội bộ nên quy chụp tuyến tính đây là tin nhắn rác.
  2. **Case #30 (Obfuscation) - "F R E E e n t r y into our weekly draw..." (Nhãn thật: spam, BoW đoán: ham):** Kẻ phát tán cố tình chèn khoảng trắng giữa các chữ cái. Bộ tách từ tokenize đơn giản phân rã chuỗi thành các token độc lập đơn lẻ: `['f', 'r', 'e', 'e']` thay vì từ khóa `free`. Các chữ cái đơn lẻ này không nằm trong từ điển vocab 2000 từ, dẫn đến vector đặc trưng bị triệt tiêu hoàn toàn về 0, làm LogReg bỏ lọt.

## 5. Thử thách cá nhân

- **Đề bài được nhận (Thử thách #4):** Chạy `knn_predict` với $k = 1, 5, 25$. Báo cáo F1 và giải thích vì sao $k$ quá nhỏ hoặc quá lớn đều có hại.
- **Kết quả đo lường thực tế (SEED = 489):**
  * **k = 1:** F1 = **0.9251** (Precision = 0.8750, Recall = 0.9813)
  * **k = 5:** F1 = **0.9251** (Precision = 0.8750, Recall = 0.9813)
  * **k = 25:** F1 = **0.9123** (Precision = 0.8595, Recall = 0.9720)
- **Giải thích:**
  * **k quá nhỏ (k = 1):** Cực kỳ nhạy cảm với các mẫu nhiễu (noise) cục bộ hoặc nhãn lỗi trong tập train. Chỉ cần 1 mẫu nhiễu nằm cực gần mẫu truy vấn sẽ lập tức quyết định sai lệch kết quả.
  * **k quá lớn (k = 25):** Vùng quyết định bị san phẳng. Trong dữ liệu lệch lớp nặng (ham chiếm đa số), số lượng lân cận quá lớn sẽ kéo theo các điểm ham xa hơn tham gia bỏ phiếu, lấn lướt và xóa nhòa các cụm phân phối spam nhỏ, làm giảm sút cả precision và recall.

## 6. Phương án cho sản phẩm thật

- **Đề xuất hệ thống:** Áp dụng mô hình **`3c. Encoder (MiniLM) + MLP head`** làm xương sống phân loại chính, kết hợp cơ chế **Router lai (Hybrid Router)**.
- **Số liệu chứng minh:**
  * `3c. Encoder + MLP head` đạt độ trễ cực thấp với **p95 = 7.42 ms** (thỏa mãn hoàn hảo SLA < 50ms), throughput đạt **1502 tin/s** trên văn bản thô, và chất lượng phân loại thuộc hàng tốt nhất hệ thống đạt **F1 = 0.981**.
  * Chuyển tiếp các tin nhắn nằm trong vùng nghi ngờ xác suất $p ∈ (0.15, 0.85)$ sang mô hình LLM nhỏ (`4a. LLM nhỏ zero-shot`) xử lý sâu.
- **Chi phí vận hành ước tính:** Chỉ tốn khoảng **10.50 USD/ngày** (~315 USD/tháng) cho lưu lượng 10 triệu tin nhắn nếu áp dụng gpt-6-luna cho các ca nghi ngờ, vô cùng tối ưu so với việc gửi toàn bộ lưu lượng qua API.
- **Đánh đổi:** Chấp nhận đầu tư chi phí phần cứng ban đầu (GPU/CPU đủ mạnh để host cục bộ encoder MiniLM phục vụ bước trích xuất embedding liên tục).

## 7. AI log và điều đã học

- **AI Log:**
  * *Phần 4:* Hỏi AI về cách cấu hình biến `reasoning_effort` phù hợp cho OpenAI và Groq. AI hướng dẫn viết wrapper mapping kwargs thông qua helper `provider_kwargs` để tắt tư duy sâu khi phân loại từ đơn. Tránh được lỗi 400 Bad Request.
  * *Phần 5:* Hỏi AI về cách xây dựng cơ chế định tuyến Hybrid Router tối ưu ranh giới phân phối. AI đề xuất chia ngưỡng xác suất tin cậy $[0.15, 0.85]$ để lọc sạch nhiễu đầu ra.
- **Điều tâm đắc nhất rút ra sau bài lab:**
  * Em hiểu ra rằng một mô hình Deep Learning phức tạp hay LLM khổng lồ không phải lúc nào cũng là viên đạn bạc giải quyết mọi vấn đề. Trong môi trường production thực tế, việc tối ưu hóa ranh giới ngữ nghĩa của không gian Embedding tĩnh kết hợp đầu phân loại tuyến tính đơn giản mang lại hiệu quả vượt trội gấp hàng nghìn lần về cả chi phí và tốc độ so với việc lạm dụng gọi API LLM liên tục.
