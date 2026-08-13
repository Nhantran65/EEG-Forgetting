# Execution Plan: Diagnostic Catastrophic Forgetting in EEG Foundation Models

Date: 2026-08-13

## Status

Active

## Outcome

Hoàn thành một submission ICASSP 2027 dài 4 trang kỹ thuật (+ trang tài liệu tham khảo nếu cần), kiểm tra câu hỏi:

> Mức chồng lấn quan trọng giữa các task EEG, đo trước chuỗi continual learning, có liên hệ với mức catastrophic forgetting về sau hay không?

Bài không đề xuất phương pháp continual-learning mới. Đóng góp chính là một chẩn đoán có kiểm chứng can thiệp:

1. Đo overlap tham số bằng Fisher importance từ các mô hình single-task độc lập, cùng khởi tạo từ CBraMod pretrained.
2. Liên hệ overlap với pairwise relative forgetting qua bốn task EEG khác nhau.
3. Kiểm chứng vị trí tham số bằng high-overlap freeze so với random-freeze tại mức hiệu năng task mới tương đương.
4. Grounding kết quả bằng một panel nhỏ channel–frequency đặc thù EEG.

Deadline mục tiêu: 2026-09-16.

## Context

- [Plan ban đầu](../../../plan%20new%20paper.md) là nguồn ý tưởng; file này thay thế nó làm execution plan hiện hành.
- Qian et al., *Learn and Don't Forget: Adding a New Language to ASR Foundation Models*, đã dùng Fisher overlap để dự báo nguy cơ quên ngôn ngữ trong Whisper. Vì phân tích của họ về bản chất đã mang tính prospective, “prospective overlap” không được xem là novelty độc lập: <https://www.isca-archive.org/interspeech_2024/qian24_interspeech.pdf>.
- EvoBrain dùng spectral affinity để điều khiển transfer. Delta của bài này là dùng overlap để chẩn đoán/giải thích forgetting và kiểm chứng bằng can thiệp, không dùng affinity để tạo một CL method mới: <https://arxiv.org/abs/2606.01767>.
- VEP public dataset: DOI <https://doi.org/10.17632/g9shp2gxhy.2>. Snapshot hiện tại có 31 subject, 59 recording sử dụng được và 3.612 cửa sổ 1 giây.
- Nguồn VEP nội bộ đã được người dùng xác nhận: `/home/kiettqa/EEG-Foundation-Model-Paper/README.md`, `paper2_benchmark/manifests/current_vep/audit.json`, `paper2_benchmark/configs/datasets/current_vep.yaml`, và manifest README cùng thư mục.
- Quy định ICASSP: 4 trang nội dung kỹ thuật, trang thứ năm chỉ dành cho references: <https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/>.

## Research Questions And Claims

### RQ chính

Overlap Fisher giữa task `i` và `j` có liên hệ với mức quên tương đối `F_rel(i <- j)` sau khi học `j` không?

Claim tối đa được phép nếu kết quả ủng hộ:

> Pre-CL parameter-importance overlap is associated with pairwise forgetting within this heterogeneous four-task EEG benchmark.

Không claim universal predictor, không claim causality chỉ từ correlation, và không claim “first”.

### RQ hỗ trợ

- Những vùng kênh–tần số nào đóng góp cho từng task?
- Tại cùng mức học task mới, freeze vùng high-overlap có bảo vệ task cũ tốt hơn random-freeze cùng layer và cùng số lượng tham số không?

### Positioning trong Intro

Thông điệp delta dự kiến:

> Prior work used Fisher overlap to anticipate language forgetting in Whisper. We investigate whether this diagnostic generalizes to heterogeneous EEG tasks, ground interference in channel–frequency structure, and validate implicated parameters through matched-plasticity interventions.

EvoBrain phải được nêu riêng: họ dùng spectral affinity để điều khiển transfer; bài này phân tích overlap như diagnostic và không đề xuất cơ chế CL mới.

## Scope

In scope:

- Một backbone duy nhất: CBraMod.
- Bốn task/dataset: BCI Competition IV-2a, PhysioNet-MI, Sleep-EDF Expanded, và VEP Mendeley.
- Main methods: sequential fine-tuning, EWC, DER++, và joint training làm upper bound.
- EvoBrain chỉ giữ nếu reproduce được trước cổng cuối Tuần 2.
- Một plasticity depth chính, chọn từ sweep 1/2/4/8 block.
- Ba task order được thiết kế để phủ cả hai chiều của mọi cặp task, ba seed.
- Occlusion là XAI chính; Integrated Gradients là kiểm tra hội tụ.
- Hai hình và một bảng trong main paper.

Out of scope:

- LaBraM và backbone thứ hai.
- Đề xuất CL algorithm mới.
- Adapter riêng theo task trong main experiment.
- Learned Conv1D channel mapping.
- Full cross-validation cho mọi method/order/seed.
- Full attribution-drift analysis D3.
- Grad-CAM, PhysioNet 64-channel, class-balanced replay và replay-method intervention, trừ khi còn thời gian.
- Cohort split giả lập thành nhiều task.

## Dataset Protocol

| Task | Main definition | Subject | Vai trò |
|---|---|---:|---|
| BCI IV-2a | 4-class motor imagery, 22 kênh | 9 | MI low-subject; cần split robustness riêng |
| PhysioNet-MI | MI protocol phải khóa đúng run/label trong Tuần 1; main dùng montage 22 kênh khớp BCI IV-2a | 109 | Cặp cùng paradigm, khác dataset |
| Sleep-EDF Expanded | 5-class sleep staging, 30 giây, bipolar | 78 | Task khác paradigm; chỉ tham gia cross-task spectral analysis |
| VEP Mendeley | 4 visual-condition labels, 14 kênh, cửa sổ 1 giây | 31 usable | Task thị giác low-density; tương phản vùng chẩm nếu được dữ liệu ủng hộ |

### VEP-specific safeguards

- Dùng subject-disjoint split; tuyệt đối không split epoch ngẫu nhiên.
- Các cửa sổ overlap 50% và các epoch cùng recording không được xem là quan sát độc lập.
- Metric và bootstrap phải aggregate/cluster theo subject; recording được giữ trọn trong subject split.
- Nhãn được suy từ thư mục recording, không phải stimulus marker. Vì vậy gọi thận trọng là “visual-condition EEG/VEP dataset”, không diễn giải như stimulus-locked ERP/VEP.
- Kiểm tra performance theo class, subject, phase và recording. Nếu model chủ yếu học recording/phase artifact hoặc kết quả không ổn định theo subject, VEP không được dùng làm bằng chứng sinh lý.
- Không mặc định attribution phải nổi bật vùng chẩm; chỉ diễn giải nếu occlusion và IG cùng ủng hộ.
- 1 recording thiếu và 4 recording preprocessing-fail phải được cố định trong manifest, không silently thay đổi giữa run.

### Shared preprocessing and channel policy

- Patch dài 1 giây; Sleep giữ epoch 30 giây dưới dạng chuỗi patch.
- Common analysis bandwidth: 0,5–40 Hz; ghi lại rõ VEP nguồn đã qua pipeline 0,1–50 Hz.
- Resample về tần số mà CBraMod checkpoint yêu cầu; kiểm tra implementation trước khi chốt 200 Hz. Không giả định nội suy 128 Hz thành 200 Hz tự tạo thêm thông tin.
- Z-score theo kênh chỉ dùng thống kê train split; không dùng test statistics.
- Dùng channel registry và canonical order cố định. Không dùng learned Conv1D để ánh xạ montage.
- Verify CBraMod downstream code nhận số kênh thay đổi. Nếu hard-code, dùng pad-and-mask vào registry chung.
- PhysioNet main dùng tập 22 kênh tương ứng BCI IV-2a; bản 64 kênh chỉ là robustness optional.
- Sleep-EDF bipolar không được chiếu thành vị trí điện cực giả. Cross-task channel-space analysis giới hạn ở montage có ý nghĩa; Sleep chỉ so trong frequency space.
- Với cửa sổ 1 giây, tránh diễn giải mạnh năng lượng rất thấp dưới khoảng 2 Hz.

## Model And Training Design

- Shared CBraMod backbone, head riêng cho từng task.
- Khi chuyển task, head cũ được đóng băng; backbone vẫn là phần có thể bị ghi đè.
- Không dùng task-specific adapter trong main experiment.
- Sweep nhanh unfreeze `1/2/4/8` block cuối bằng sequential FT trên một order và một seed.
- Đồng thời chạy single-task baselines cho cả bốn task. Chọn một depth duy nhất: depth nhỏ nhất đạt ít nhất 95% best mean normalized validation BA qua bốn task. Không chọn depth dựa trên mức forgetting.
- Khóa common preprocessing, optimizer family, schedule và validation budget sau pilot.
- Method-specific hyperparameter được tune với cùng validation budget; không tune lại theo order/seed.

Main methods:

1. Sequential FT: baseline gây quên và nền sạch cho causal intervention.
2. EWC: regularization baseline; memory accounting gồm Fisher và `theta*` cho toàn bộ tham số plastic.
3. DER++: replay baseline, giữ reservoir policy gốc.
4. Joint training: upper bound, không đưa vào diễn giải như continual method.
5. EvoBrain: optional; bỏ khỏi experiment nếu chưa reproduce đúng trước cuối Tuần 2.

### Replay fairness

- Equalize replay bằng persistent-memory byte cap, không bằng số item.
- Buffer accounting gồm EEG samples, label, logits/responses và metadata cần thiết.
- Báo thêm số item và tổng thời lượng EEG để người đọc hiểu ngân sách.
- Giữ sampling policy nguyên bản của mỗi method. Class-balanced replay chỉ là optional ablation.
- Chọn byte cap bằng một pilot trên một cặp task rồi khóa cho mọi cặp/order.
- Bảng memory phải báo total method footprint, gồm cả EWC state, không chỉ replay buffer.

## Measurements

### Continual-learning performance

Lưu performance theo subject:

`R[i,j,s]` = balanced accuracy của subject `s` thuộc task `j`, đo sau khi học xong task `i`.

- Ma trận `R` chính dùng balanced accuracy.
- AA, BWT và Average Forgetting đều dẫn xuất từ `R` này.
- Accuracy thường là ma trận phụ để đối chiếu prior work.
- Macro-F1 là metric hỗ trợ; Cohen's kappa bắt buộc cho Sleep-EDF.
- Bỏ FWT vì head task mới chưa train làm metric chủ yếu phản ánh random head.

Pairwise forgetting do task `j` gây lên task cũ `i`:

`F(i <- j) = R[before j, i] - R[after j, i]`

Metric chính chuẩn hóa theo headroom:

`F_rel(i <- j) = F(i <- j) / (R[before j, i] - chance_i)`

- Báo `F` raw như sensitivity analysis.
- Flag hoặc loại theo quy tắc predeclared nếu mẫu số gần chance; không âm thầm clip.
- Lưu prediction và metric per subject. Đây là repeated measurement, không biến subject thành task-pair độc lập.

### Prospective task overlap

- Train single-task models độc lập từ cùng pretrained CBraMod, cùng selected depth và protocol.
- Tính diagonal empirical Fisher importance trên shared-backbone parameters; không đưa task head vào overlap.
- Normalize importance theo layer trước khi ghép để layer lớn không tự động chi phối.
- Primary overlap: cosine similarity giữa hai Fisher-importance vector.
- Sensitivity: top-k Jaccard và layer-wise overlap; `k` được predeclare từ validation, không chọn sau khi thấy correlation.
- Instrument check: overlap cùng task qua seed/split phải ổn định và cao hơn rõ rệt overlap cross-task. Nếu không, dừng CL scale-up và sửa thước đo.
- Qian et al. đã có prospective Fisher-overlap logic trong ASR; novelty không đặt ở riêng phép đo này.

### Channel–frequency grounding

- Primary: band/channel occlusion và mức giảm BA.
- Validation: Integrated Gradients; so agreement theo rank, không yêu cầu trị tuyệt đối giống nhau.
- Bands được predeclare trong 0,5–40 Hz; phương thức loại band phải giống nhau giữa task.
- Cross-dataset spatial map dùng canonical anatomical regions/mask cho kênh thiếu; Sleep không tham gia spatial comparison.
- Panel này dùng để định vị bài là EEG research, không gánh claim chính và chiếm tối đa khoảng 15% diện tích Hình 1.

## Analysis Design

- Bốn task tạo 6 cặp không hướng và tối đa 12 cặp có hướng.
- Chọn ba order sao cho một order và reverse order phủ cả hai chiều của mọi cặp; order thứ ba cân bằng/challenging nhưng phải predeclare trước full run.
- Primary visual: overlap vs subject-aggregated `F_rel`, hiển thị task pair, direction, method và uncertainty.
- Hierarchical analysis giữ subject nested trong dataset và order/seed là repeated/random effects thích hợp.
- Vì overlap chỉ thay đổi ở cấp task-pair, không dùng số subject để giả vờ tăng số task-pair độc lập. P-value là exploratory; báo effect size, interval và leave-one-pair-out sensitivity.
- Kiểm tra xu hướng theo layer/depth; không chọn layer sau khi nhìn kết quả mà không ghi rõ exploratory.
- Kết luận RQ2 phải ổn định về dấu và không do một cặp task duy nhất chi phối.

## Causal Diagnostic Validation

Chạy trước trên sequential FT, ít nhất một cặp task hoàn thành trước cuối Tuần 3.

1. Xếp hạng shared-backbone units/parameter groups theo overlap importance.
2. Freeze một số tỷ lệ predeclared của high-overlap units khi học task mới.
3. Control: random-freeze cùng layer, cùng số units/parameters; lặp ít nhất ba random masks.
4. Đo đồng thời old-task forgetting và new-task BA.
5. So sánh stability–plasticity Pareto hoặc nội suy tại matched new-task performance.

Claim can thiệp chỉ được dùng khi:

> At matched new-task performance, freezing high-overlap parameters reduces forgetting more than freezing an equal number of randomly selected parameters in the same layers.

Nếu còn ngân sách, xác nhận trên DER++; không cần chạy mọi method.

## Paper Budget

- Intro: khoảng 0,6 trang.
- Related work: khoảng 0,3 trang.
- Method: khoảng 0,9 trang.
- Experimental setup: khoảng 0,6 trang.
- Results: khoảng 1,3 trang.
- Conclusion: khoảng 0,2 trang.

Main artifacts:

- Figure 1: overlap vs `F_rel`, kèm panel channel–frequency nhỏ.
- Figure 2: stability–plasticity Pareto hoặc matched-performance causal comparison; đây là headline figure.
- Table 1: task performance, BWT và total memory footprint.

Không thêm hình thứ ba vào main paper.

## Approach And Schedule

### Week 1 — 2026-08-13 to 2026-08-19: protocol and instrument

- Khóa task definition, label mapping và split manifest cho cả bốn dataset.
- Audit VEP theo subject/recording/phase/class và xác nhận không leakage.
- Gửi yêu cầu truy cập TUEV ngay đầu tuần chỉ làm fallback bảo hiểm.
- Verify CBraMod checkpoint, variable-channel behavior, patching và preprocessing.
- Chạy linear probe và single-task FT baselines.
- Chạy sweep FT 1/2/4/8 block nhanh; chọn một depth chính bằng validation rule.
- Tạo Fisher-overlap signatures và kiểm tra same-task reproducibility.
- Chạy sequential FT smoke test end-to-end.

Gate cuối Tuần 1:

- Performance hợp lý so với reference có cùng protocol, không dùng một ngưỡng công bố sai protocol.
- Fine-tune phải tốt hơn linear probe đủ rõ để chứng minh phần plastic thực sự học.
- Fisher overlap phải có độ phân giải: same-task qua seed/split ổn định và tách được cross-task.
- VEP phải qua audit leakage/confound và có performance subject-disjoint ổn định. Nếu fail, kích hoạt TUEV ngay; không chờ tới Tuần 3.

### Week 2 — 2026-08-20 to 2026-08-26: continual-learning matrix

- Chạy FT, EWC, DER++ với 3 order x 3 seed ở selected depth.
- Log `R[i,j,s]`, predictions, checkpoints và memory footprint.
- Chạy split robustness thứ hai cho BCI IV-2a sớm.
- Reproduce EvoBrain với deadline cứng.

Gate giữa Tuần 2:

- Median `F_rel` phải vượt seed noise ở đa số directed transitions tại ít nhất một plasticity depth.
- Dấu kết luận không được đảo hoàn toàn trên split BCI thứ hai.
- Nếu fail: tăng depth đã predeclared hoặc dùng challenging order; không thêm task mới lúc này.

Gate cuối Tuần 2:

- EvoBrain reproduce đúng thì giữ; nếu không, bỏ khỏi experiments và chuyển sang Related Work.

### Week 3 — 2026-08-27 to 2026-09-02: diagnostic and intervention

- Hoàn tất overlap extraction và pairwise `F_rel` dataset.
- Chạy occlusion; IG chỉ để rank-convergence validation.
- Hoàn thành causal intervention trên ít nhất một cặp với sequential FT.
- Bắt đầu Figure 1, Figure 2 và draft Method/Setup.

Gate cuối Tuần 3:

- Occlusion và IG phải đồng thuận ở mức rank đủ để dùng panel EEG.
- Causal intervention phải hoàn thành trên ít nhất một cặp.
- Nếu thiếu thời gian, cắt breadth của channel–frequency/IG và mọi D3 còn lại; không cắt intervention.

### Week 4 — 2026-09-03 to 2026-09-09: analysis and full draft

- Fit hierarchical/descriptive analyses, uncertainty và leave-one-pair-out checks.
- Hoàn thiện 2 figures + 1 table.
- Viết full four-page draft; nêu Qian và EvoBrain delta trực tiếp.
- Internal review tập trung claim strength, leakage và page budget.

Gate cuối Tuần 4:

- Nếu RQ2 ổn định qua layer/condition: dùng cautious association claim.
- Nếu RQ2 yếu/null: báo null có giới hạn trong four-task benchmark; giữ causal intervention làm headline nếu nó thành công.
- Nếu cả association và intervention đều fail: không claim predictor; đánh giá chuyển journal hoặc reframing trước khi nộp.

### Submission buffer — 2026-09-10 to 2026-09-16

- Sửa paper, kiểm tra reproducibility, references, ICASSP format và supplementary repository.
- Freeze số liệu/hình trước deadline ít nhất 48 giờ nếu có thể.

## Risks And Recovery

- **Novelty overlap với Qian:** cite trực tiếp; novelty dựa vào EEG grounding + systematic sequential benchmark + matched causal intervention.
- **VEP recording-level confound:** subject/recording audit, conservative naming và no physiological claim nếu evidence không hội tụ; TUEV là fallback chỉ khi được kích hoạt trong Tuần 1.
- **BCI IV-2a chỉ 9 subject:** per-subject logging và split robustness sớm; không giả vờ tăng power bằng epoch count.
- **Overlap có ít independent task pairs:** effect size, interval, leave-one-pair-out và cautious claim; không quảng bá universal prediction.
- **Forgetting quá nhỏ:** plasticity-depth sweep và challenging order đã predeclare.
- **EvoBrain tốn thời gian:** deadline cứng cuối Tuần 2, sau đó drop.
- **Page overflow:** 2 figures + 1 table; D3/Grad-CAM/extra montage để supplementary hoặc bỏ.
- **Compute failure:** ưu tiên FT -> causal intervention -> EWC/DER++ -> XAI breadth -> EvoBrain.
- **RQ2 null:** báo đúng giới hạn; không đổi metric/layer hậu nghiệm để săn correlation.

Recovery is non-destructive: giữ mọi manifest, config, checkpoint và result đã hoàn thành; thay dataset/method chỉ bằng config/version mới, không ghi đè run cũ.

## Progress

- [x] Chốt diagnostic/XAI scope và CBraMod-only backbone.
- [x] Chốt bốn dataset; VEP là task thứ tư, TUEV chỉ fallback.
- [x] Chốt main methods, metrics, memory accounting và page budget.
- [x] Đọc và định vị novelty so với Qian et al. và EvoBrain.
- [x] Chốt VEP subject count, montage, preprocessing và các hạn chế.
- [ ] Khóa exact PhysioNet-MI runs/labels và Sleep-EDF subset/protocol.
- [ ] Tạo immutable dataset/split manifests.
- [ ] Hoàn thành Week 1 gate.
- [ ] Hoàn thành Week 2 CL matrix.
- [ ] Hoàn thành Week 3 diagnostic/intervention gate.
- [ ] Hoàn thành Week 4 analysis/full draft gate.
- [ ] Hoàn thành submission package và validation.

## Decisions

- 2026-08-13: Bài là diagnostic/XAI paper, không đề xuất continual-learning method mới.
- 2026-08-13: CBraMod là backbone duy nhất; shared backbone + task-specific heads; không adapter trong main.
- 2026-08-13: Unfreeze-depth sweep 1/2/4/8 là pilot; main dùng một depth chọn bằng single-task validation, không chọn theo forgetting.
- 2026-08-13: Bốn task chính là BCI IV-2a, PhysioNet-MI, Sleep-EDF và VEP; không cohort-split.
- 2026-08-13: VEP được giữ vì có 31 usable subjects và tạo tương phản thị giác/low-density; phải diễn giải thận trọng do label recording-level và window overlap.
- 2026-08-13: TUEV chỉ là fallback được kích hoạt trong Tuần 1 nếu VEP fail audit.
- 2026-08-13: Main `R` dùng balanced accuracy; pairwise outcome chính là `F_rel`; bỏ FWT.
- 2026-08-13: Replay equalized theo persistent bytes; EWC memory gồm Fisher + `theta*`; DER++ giữ reservoir gốc.
- 2026-08-13: Prospective Fisher overlap không phải novelty độc lập vì Qian et al. đã có logic gần tương đương trong Whisper.
- 2026-08-13: Can thiệp high-overlap vs matched random-freeze tại matched new-task performance là bằng chứng mạnh nhất.
- 2026-08-13: Main paper giới hạn 2 figures + 1 table; channel–frequency chỉ là panel nhỏ để EEG grounding.

## Validation

- Focused proof:
  - Unit tests cho `R`, AA, BWT, AF, `F`, `F_rel`, near-chance handling và memory accounting.
  - Dataset audit: unique subject across split; recording integrity; VEP missing/fail manifest; class/age distribution.
  - Shape/mask smoke tests cho 2/14/22-channel inputs và 30-second Sleep sequence.
  - Deterministic same-task Fisher reproducibility check.
- Integration proof:
  - Một four-task sequential FT run end-to-end, gồm checkpoint, `R[i,j,s]`, overlap extraction và intervention mask.
  - Recompute main table/figures từ frozen result artifacts bằng một command documented trong repository.
- Statistical proof:
  - Subject-clustered uncertainty, order/seed handling, leave-one-pair-out sensitivity và raw-`F` sensitivity.
  - Matched-performance/Pareto comparison cho high-overlap versus random-freeze.
- Repository-required checks:
  - Plan progress và decisions được cập nhật sau mỗi gate.
  - Không overwrite manifest/config/result đã dùng trong paper.
  - Final manuscript numbers trace được về immutable run IDs.

## Result

Chưa hoàn thành. Plan được tạo sau khi scope, novelty positioning và VEP viability đã được chốt. Việc tiếp theo là khóa exact dataset protocols và chạy Week 1 gate.
