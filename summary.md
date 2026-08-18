
### Paper summary

Paper nghiên cứu:

> Khi EEG foundation model học thêm task mới, nó có còn sử dụng cùng kênh EEG và dải tần để giải quyết task cũ không?

Thí nghiệm dùng CBraMod trên BCI IV-2a, High-Gamma và Sleep-EDF, so sánh Sequential FT, EWC và DER++. Channel–frequency reliance được đo bằng spectral occlusion và classification-margin drop. Độ thay đổi explanation được đo bằng noise-corrected Physiological Explanation Drift — PED.

### Research questions

- **RQ1 — Explanation drift:** Bằng chứng channel–frequency của task cũ thay đổi bao nhiêu sau khi model học task mới?
- **RQ2 — Performance–explanation alignment:** Phương pháp giảm performance forgetting có đồng thời bảo tồn explanation không? Accuracy ổn định có đủ để kết luận cách model ra quyết định ổn định không?

### Contributions

1. Đề xuất protocol đo channel–frequency reliance trong continual EEG bằng spectral occlusion và margin drop.
2. Xây dựng PED có hiệu chỉnh measurement noise bằng split-half cross-minus-within JSD.
3. Đưa reliability và fidelity thành hard gates trước khi diễn giải XAI.
4. So sánh explanation retention của Sequential FT, EWC và DER++ trên cùng checkpoint, subject và sample.
5. Chứng minh accuracy retention không phải đại diện đáng tin cậy cho explanation stability.

### Kết quả chính

- Sequential FT tạo PED lớn, kể cả khi accuracy không giảm hoặc còn tăng.
- EWC và DER++ giảm PED so với Sequential ở **21/21** matched comparisons.
- EWC giữ explanation tốt nhất.
- Tương quan giữa forgetting và PED yếu, xác nhận performance và explanation là hai chiều đánh giá khác nhau.

### Vị trí trong literature

Prior work chủ yếu nghiên cứu:

- Continual EEG dựa trên accuracy/forgetting.
- EEG explainability tại một checkpoint tĩnh.
- Explanation drift trong continual learning nói chung.

Paper này nối ba hướng trên bằng cách nghiên cứu **lifetime stability của bằng chứng sinh lý channel–frequency trong continual EEG foundation models**.

Paper không đề xuất CL algorithm mới và không xem attribution map là bằng chứng nhân quả sinh học.
