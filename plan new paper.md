Rồi, đây là plan chạy experiment hoàn chỉnh cho hướng **diagnostic/XAI** — nhớ nguyên tắc lõi: em **không build cơ chế chống quên**, em build **công cụ đo và giải thích việc quên**, rồi áp lên các phương pháp CL có sẵn (kể cả EvoBrain). Toàn bộ thiết kế xoay quanh việc tạo ra 3 artifact cho bài: **bản đồ engram**, **tương quan overlap–forgetting**, **phân bố forgetting theo sinh lý**.

## Phần A — Định nghĩa "engram" cho tính toán (làm rõ trước khi code)

Đây là thứ phải chốt đầu tiên, vì mọi thứ dựa vào nó. Em định nghĩa engram ở **hai không gian**, mỗi cái phục vụ một RQ:

- **Engram không gian đầu vào (channel × băng tần)** — dùng cho RQ1 & RQ3. Trả lời "task này dựa vào kênh/băng tần nào". Đây là phần neuro-interpretable mà EvoBrain không có.
- **Engram không gian tham số (units quan trọng)** — dùng cho RQ2. Trả lời "overlap giữa task → interference → forgetting".

## Phần B — Hạ tầng & data (Tuần 1)

- **Backbone**: CBraMod làm chính (checkpoint mở), LaBraM để robustness check nếu còn thời gian. Đóng băng backbone + head/adapter nhẹ mỗi task.
- **3 dataset**: BCI IV-2a (MI, 22ch) · Sleep-EDF Expanded (sleep, 2ch) · VEP Mendeley của em (thị giác). Ba paradigm + ba montage = forgetting rõ.
- **Tiền xử lý đồng nhất**: resample 200Hz, patch 1s, z-score theo kênh, band-pass 0.5–40Hz. Conv1D channel-mapping để khớp số kênh với checkpoint. Split **theo subject**, không rò rỉ.
- **Tool**: MOABB/braindecode (data loader), MNE (lọc băng tần), Captum (attribution), PyTorch. Xin quota LUMI.

## Phần C — Chạy các phương pháp CL (đối tượng phân tích, không phải đối thủ) (Tuần 2)

Đây là "chất nền" để em soi. Chạy backbone qua chuỗi task tuần tự với từng phương pháp:

- **sequential FT** (đáy, quên nhiều nhất)
- **EWC** (bonus: cho luôn Fisher importance — dùng miễn phí cho Phần D)
- **DER++**
- **EvoBrain** (reimplement — ở đây là *đối tượng phân tích*, không cần thắng)
- **joint multi-task** (trần)

Cấu hình bắt buộc để chắc số:
- **2–3 thứ tự task khác nhau** (forgetting phụ thuộc thứ tự) — vd A→B→C, C→A→B, B→C→A.
- **3–5 seed** để có khoảng tin cậy.
- Log **per-task accuracy sau mỗi bước** → tính Backward Transfer / Average Forgetting / Forward Transfer.

**→ Cổng quyết định cuối Tuần 2:** forgetting phải đo được và đủ lớn. Nếu các phương pháp đã gần như không quên → tăng độ khó (thêm dataset thứ 4 PhysioNet-MI, hoặc thứ tự khắc nghiệt hơn). Nếu forgetting quá nhỏ trên mọi phương pháp thì premise bài yếu → báo Jenni sớm.

## Phần D — Pipeline chẩn đoán (đây mới là contribution) (Tuần 3)

Ba khối, chạy trên **mọi checkpoint đã lưu** ở Phần C:

**D1 — Bản đồ engram channel × băng tần (cho RQ1, RQ3)**
- **Attribution gradient** (Captum: Integrated Gradients + Grad-CAM — đúng chuyên môn em) trên input channel×time cho mỗi task.
- **Occlusion kiểm chứng**: zero-out từng kênh / từng băng tần (δ θ α β γ), đo mức tụt accuracy → "importance" của kênh/băng đó. Robust, dễ diễn giải, làm bằng chứng hội tụ với gradient.
- Gộp thành heatmap **kênh × băng tần** cho mỗi (task, phương pháp). Đây là "engram map".

**D2 — Importance tham số & overlap (cho RQ2)**
- Với mỗi task, lấy vector importance của units/params (Fisher từ EWC có sẵn, hoặc attribution tới hidden units).
- **Overlap giữa task i và j** = Jaccard của top-k units quan trọng, hoặc cosine của vector importance.

**D3 — Chỉ số forgetting theo sinh lý**
- Đo **dịch chuyển attribution** của một task sau khi học task sau: engram của nó ở băng tần/kênh nào bị "ghi đè" nhiều nhất.

## Phần E — Ba phân tích ra ba artifact của bài (Tuần 4)

- **RQ1 → Hình 1 (localization):** engram map cho từng task, chỉ ra task-knowledge cư trú ở đâu. Kèm nhận xét sinh lý (vd MI dựa β/μ vùng vận động, sleep dựa δ/θ, VEP dựa vùng chẩm).
- **RQ2 → Hình 2 (headline):** scatter **overlap engram vs forgetting**, tính **Spearman/Pearson**, làm **across cả 4 phương pháp CL kể cả EvoBrain**. Đây là kết quả bán bài.
- **RQ3 → Hình 3 + Bảng 1:** forgetting tập trung ở băng tần/kênh nào (phân bố), và bảng gợi ý "cặp task overlap cao ở băng X → nên dùng phương pháp Y". Đây là giá trị thực dụng.

**→ Cổng quyết định cuối Tuần 4:** RQ2 phải có tín hiệu. Nếu overlap **có** dự báo forgetting → bài mạnh. Nếu **không** (null) → vẫn nộp được nhưng đóng khung lại là "phát hiện EEG FM khác vision: overlap không dự báo forgetting" — vẫn là finding, chỉ cần viết đúng.

## Phần F — Kiểm chứng & ablation (xen Tuần 3–4)

- **Hội tụ attribution**: gradient vs occlusion phải cho engram map tương tự → tăng độ tin.
- **Ổn định qua seed/thứ tự**: engram map và tương quan phải giữ được, không phải artefact một lần chạy.
- **Sanity check**: engram của hai task cùng paradigm (nếu thêm PhysioNet-MI) phải overlap cao hơn hai task khác paradigm.
- *(Tùy chọn stretch — chỉ nếu dư thời gian):* đóng băng thử các units overlap cao xem forgetting có giảm không. **Cẩn thận**: cái này bắt đầu giống "method" → nếu làm, đóng khung là *validation cho công cụ chẩn đoán*, không phải phương pháp đề xuất, để khỏi chạm EvoBrain.

## Phần G — Viết & nộp (Tuần 4.5–5)

Intro (positioning nêu đích danh EvoBrain NSN/RAD, tuyên bố rõ "không đề xuất CL method, cung cấp explainability framework") → Method (pipeline D1–D3, 1 hình) → Experiments (Hình 1+2+3, Bảng 1) → Conclusion. Jenni review, chỉnh format 4+1 trang template ICASSP, nộp trước **16/9/2026**.

## Lịch nén (hôm nay 13/8 → deadline 16/9, ~4.5 tuần, phải bắt đầu tuần này)

| Tuần | Việc | Ra được gì |
|---|---|---|
| 1 | Hạ tầng + data + reproduce 1 CL run | forgetting number đầu tiên |
| 2 | Chạy đủ 5 phương pháp × 2–3 order × seed | bảng forgetting (nền phân tích) |
| 3 | Build pipeline D1–D3, sinh engram map | công cụ chẩn đoán chạy được |
| 4 | 3 phân tích + hình/bảng + ablation | Hình 1/2/3, Bảng 1 |
| 4.5–5 | Viết + review Jenni + polish + nộp | bản 4 trang submit |

**Ba rủi ro phải canh:** (1) forgetting quá nhỏ → tăng độ khó (cổng T2); (2) RQ2 null → đổi khung thành finding, đừng giấu; (3) thời gian — nếu T3 chưa có pipeline chạy, chuyển sang **journal TNSRE/JBHI** giữ nguyên toàn bộ đồ đã làm, không phí.

**Điểm mạnh của plan này:** không phụ thuộc việc phải thắng SOTA, dùng đúng thế mạnh XAI của em, và mỗi tuần đều ra một artifact cụ thể để khoe Jenni — nên kể cả không kịp ICASSP thì em vẫn có một bài journal hoàn chỉnh trong tay.

Muốn mình gói plan này thành **file checklist (Markdown hoặc Word) chia theo tuần có ô tick** để em bám khi chạy, hay dựng luôn **skeleton code repo** (cấu trúc thư mục + data loader + vòng continual + chỗ cắm attribution) để em bắt tay tuần này?

=============================================
=============================================
=============================================


# ICASSP 2027 – Experiment Checklist

**Bài:** Diagnostic/XAI framework để định vị và dự báo catastrophic forgetting trong EEG foundation model
**Deadline nộp:** 16/9/2026 (AoE) · **Hội:** ICASSP 2027, Toronto
**Nguyên tắc lõi:** KHÔNG build cơ chế chống quên. Build công cụ ĐO và GIẢI THÍCH việc quên, rồi áp lên các phương pháp CL có sẵn (kể cả EvoBrain).

**3 artifact phải ra:** (1) engram map channel×band · (2) tương quan overlap–forgetting · (3) phân bố forgetting theo sinh lý

---

## Chốt trước khi code – Định nghĩa engram

- [ ] Engram không gian đầu vào (channel × băng tần) — phục vụ RQ1 & RQ3
- [ ] Engram không gian tham số (units quan trọng) — phục vụ RQ2
- [ ] Viết 1 đoạn định nghĩa toán học gọn cho cả hai (để dán vào Method)

---

## Tuần 1 – Hạ tầng & data

- [ ] Lấy checkpoint CBraMod (HuggingFace) làm backbone chính
- [ ] (Tùy chọn) Lấy LaBraM để robustness check nếu dư thời gian
- [ ] Tải BCI Competition IV-2a (MI, 22ch)
- [ ] Tải Sleep-EDF Expanded (sleep, 2ch)
- [ ] Chuẩn bị VEP Mendeley 4-class của mình
- [ ] Cài môi trường: PyTorch, MOABB, braindecode, MNE, Captum
- [ ] Xin quota LUMI, test chạy được job GPU
- [ ] Tiền xử lý đồng nhất: resample 200Hz, patch 1s, z-score theo kênh, band-pass 0.5–40Hz
- [ ] Conv1D channel-mapping để khớp số kênh (22/2/VEP) với input checkpoint
- [ ] Split train/val/test THEO SUBJECT (kiểm tra không rò rỉ)
- [ ] Reproduce 1 continual run đơn giản (sequential FT trên 3 task)

**Ra được:** forgetting number đầu tiên, pipeline chạy end-to-end

---

## Tuần 2 – Chạy các phương pháp CL (đối tượng phân tích)

- [ ] sequential FT (đáy)
- [ ] EWC (lưu luôn Fisher importance để tái dùng ở Tuần 3)
- [ ] DER++
- [ ] EvoBrain (reimplement — ở đây là đối tượng phân tích, KHÔNG cần thắng)
- [ ] joint multi-task (trần)
- [ ] Chạy 2–3 thứ tự task khác nhau (A→B→C, C→A→B, B→C→A)
- [ ] Chạy 3–5 seed mỗi cấu hình
- [ ] Log per-task accuracy sau MỖI bước
- [ ] Tính Average Accuracy, Backward Transfer / Forgetting, Forward Transfer
- [ ] Lưu toàn bộ checkpoint trung gian (cần cho Tuần 3)

### ⛔ Cổng quyết định cuối Tuần 2
- [ ] Forgetting đo được và ĐỦ LỚN?
  - Nếu quá nhỏ trên mọi phương pháp → thêm dataset thứ 4 (PhysioNet-MI) hoặc thứ tự khắc nghiệt hơn
  - Nếu vẫn không có forgetting → premise yếu, báo Jenni sớm

**Ra được:** bảng forgetting đầy đủ (nền cho mọi phân tích)

---

## Tuần 3 – Pipeline chẩn đoán (đây là contribution)

Chạy trên MỌI checkpoint đã lưu ở Tuần 2.

### D1 – Engram map channel × băng tần (RQ1, RQ3)
- [ ] Attribution gradient: Integrated Gradients + Grad-CAM (Captum) trên input channel×time mỗi task
- [ ] Occlusion: zero-out từng kênh / từng băng tần (δ θ α β γ), đo mức tụt accuracy
- [ ] Gộp thành heatmap kênh × băng tần cho mỗi (task, phương pháp)

### D2 – Importance tham số & overlap (RQ2)
- [ ] Lấy vector importance của units/params mỗi task (Fisher từ EWC hoặc attribution tới hidden units)
- [ ] Tính overlap giữa task i và j (Jaccard top-k units hoặc cosine importance)

### D3 – Chỉ số forgetting theo sinh lý
- [ ] Đo dịch chuyển attribution của một task sau khi học task kế
- [ ] Xác định băng tần/kênh bị ghi đè nhiều nhất

**Ra được:** công cụ chẩn đoán chạy được trên tất cả phương pháp

---

## Tuần 4 – Ba phân tích → ba artifact

- [ ] **RQ1 → Hình 1 (localization):** engram map từng task + nhận xét sinh lý (MI dựa β/μ vận động, sleep dựa δ/θ, VEP dựa vùng chẩm)
- [ ] **RQ2 → Hình 2 (headline):** scatter overlap vs forgetting, tính Spearman/Pearson, làm across cả 4 phương pháp kể cả EvoBrain
- [ ] **RQ3 → Hình 3 + Bảng 1:** phân bố forgetting theo băng tần/kênh + bảng gợi ý "cặp task overlap cao ở băng X → dùng phương pháp Y"

### Kiểm chứng & ablation (xen Tuần 3–4)
- [ ] Hội tụ attribution: gradient vs occlusion cho engram map tương tự
- [ ] Ổn định qua seed/thứ tự: engram map và tương quan giữ được
- [ ] Sanity check: 2 task cùng paradigm overlap cao hơn khác paradigm
- [ ] (Stretch, tùy chọn) đóng băng units overlap cao xem forgetting có giảm — nếu làm, đóng khung là VALIDATION cho công cụ, KHÔNG phải method đề xuất

### ⛔ Cổng quyết định cuối Tuần 4
- [ ] RQ2 có tín hiệu?
  - Có → bài mạnh
  - Null → vẫn nộp, đổi khung thành "EEG FM khác vision: overlap không dự báo forgetting". Là finding, viết đúng, đừng giấu.

**Ra được:** Hình 1/2/3 + Bảng 1

---

## Tuần 4.5–5 – Viết & nộp

- [ ] Intro: positioning nêu đích danh EvoBrain (NSN/RAD), tuyên bố rõ "không đề xuất CL method, cung cấp explainability framework"
- [ ] Method: pipeline D1–D3 + 1 hình kiến trúc
- [ ] Experiments: Hình 1 + 2 + 3 + Bảng 1
- [ ] Conclusion
- [ ] Bám đúng 3 contribution diagnostic (localization + overlap-predicts-forgetting + practical tool)
- [ ] Kiểm tra KHÔNG claim "first" / "cơ chế CL mới"
- [ ] Jenni review vòng 1
- [ ] Sửa theo comment Jenni
- [ ] Format đúng template ICASSP 4+1 trang
- [ ] Nộp trước 16/9/2026

---

## Đường lui (nếu tuần 3 thấy đuối)

- [ ] Giữ nguyên toàn bộ data + kết quả đã làm
- [ ] Bỏ áp lực 4 trang, mở rộng thành journal IEEE TNSRE hoặc JBHI
- [ ] Thêm MEG + XAI đầy đủ khi có thời gian / dữ liệu CIBR
- [ ] Không phí gì đã làm

---

## Ba rủi ro canh suốt

1. Forgetting quá nhỏ → tăng độ khó (cổng T2)
2. RQ2 null → đổi khung thành finding, đừng giấu
3. Thời gian → nếu T3 chưa có pipeline, chuyển journal, giữ nguyên đồ đã làm

---

## Nhắc nhanh về data

| Dataset | Paradigm | Kênh | Vai trò |
|---|---|---|---|
| BCI IV-2a | Motor imagery (4-class) | 22 | task 1 |
| Sleep-EDF Expanded | Sleep staging (5-class) | 2 | task 2, tạo shift montage mạnh |
| VEP Mendeley (của mình) | Visual (4-class) | — | task 3, gắn vào mạch nghiên cứu riêng |
| PhysioNet MI | Motor imagery | 64 | tùy chọn task 4, test cross-montage cùng paradigm |

Tránh SEED/DEAP (emotion) vì phải xin license, dễ mất 1–2 tuần.
