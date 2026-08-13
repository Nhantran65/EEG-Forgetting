# Execution Plan: Diagnostic Catastrophic Forgetting in EEG Foundation Models

Date: 2026-08-13

## Status

Active

## Outcome

Hoàn thành một submission ICASSP 2027 dài 4 trang kỹ thuật (+ trang tài liệu tham khảo nếu cần), kiểm tra câu hỏi:

> Mức chồng lấn quan trọng giữa các task EEG, đo trước chuỗi continual learning, có liên hệ với mức catastrophic forgetting về sau hay không?

Bài không đề xuất phương pháp continual-learning mới. Đóng góp chính là một chẩn đoán có kiểm chứng can thiệp:

1. Đo overlap tham số bằng Fisher importance từ các mô hình single-task độc lập, cùng khởi tạo từ CBraMod pretrained.
2. Liên hệ overlap với pairwise relative forgetting qua ba task EEG đã có và TUEV nếu quyền truy cập hoàn tất đúng hạn.
3. Kiểm chứng vị trí tham số bằng high-overlap freeze so với random-freeze tại mức hiệu năng task mới tương đương.
4. Grounding kết quả bằng một panel nhỏ channel–frequency đặc thù EEG.

Deadline mục tiêu: 2026-09-16.

## Context

- [Plan ban đầu](../../../plan%20new%20paper.md) là nguồn ý tưởng; file này thay thế nó làm execution plan hiện hành.
- Qian et al., *Learn and Don't Forget: Adding a New Language to ASR Foundation Models*, đã dùng Fisher overlap để dự báo nguy cơ quên ngôn ngữ trong Whisper. Vì phân tích của họ về bản chất đã mang tính prospective, “prospective overlap” không được xem là novelty độc lập: <https://www.isca-archive.org/interspeech_2024/qian24_interspeech.pdf>.
- EvoBrain dùng spectral affinity để điều khiển transfer. Delta của bài này là dùng overlap để chẩn đoán/giải thích forgetting và kiểm chứng bằng can thiệp, không dùng affinity để tạo một CL method mới: <https://arxiv.org/abs/2606.01767>.
- Audit trực tiếp VEP cho thấy 0 marker trong 64 JSON, 0 annotation trong 63 EDF, và phase-crossover BA xấp xỉ chance 0,25. VEP bị quarantine khỏi RQ2, CL matrix và physiological claims; manifest/adapter cũ vẫn được giữ để audit paper trước.
- TUEV là task thứ tư có điều kiện. Preprocessing phải tái dùng đúng code CBraMod commit `b9e961003214326972c567eff390e75b0287e32a`; output thực tế là 16 TCP bipolar channel x 5 patch x 200 điểm, không phải 23 channel model input.
- Ba dataset công khai đã được materialize dưới `data/raw/`: BCI IV-2a có 18 GDF + official true labels, PhysioNet-MI có 109 x 6 imagery-run EDF, Sleep Cassette có 153 cặp PSG/hypnogram trên 78 subject. Checksum/ZIP CRC và source-level annotation audit đều pass.
- Quy định ICASSP: 4 trang nội dung kỹ thuật, trang thứ năm chỉ dành cho references: <https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/>.

## Research Questions And Claims

### RQ chính

Overlap Fisher giữa task `i` và `j` có liên hệ với mức quên tương đối `F_rel(i <- j)` sau khi học `j` không?

Claim tối đa được phép nếu kết quả ủng hộ:

> Pre-CL parameter-importance overlap is associated with pairwise forgetting within this heterogeneous EEG benchmark.

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
- Ba task chắc chắn: BCI Competition IV-2a, PhysioNet-MI và Sleep-EDF Expanded; TUEV là task thứ tư nếu có dữ liệu chạy được trước 2026-08-19.
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
- VEP trong main RQ2, continual-learning matrix hoặc physiological claims.

## Dataset Protocol

| Task | Main definition | Subject | Vai trò |
|---|---|---:|---|
| BCI IV-2a | 4-class motor imagery, 22 kênh | 9 | MI low-subject; cần split robustness riêng |
| PhysioNet-MI | Runs 04/06/08/10/12/14, bỏ T0, 4-class theo run; main dùng 22 kênh khớp BCI IV-2a; clean split giữ 105 subject | 105 main / 109 reproduction | Cặp cùng paradigm, khác dataset |
| Sleep-EDF Expanded | 5-class sleep staging, 30 giây, bipolar | 78 | Task khác paradigm; chỉ tham gia cross-task spectral analysis |
| TUEV v2.0.0 | 6-class event classification, 16 TCP bipolar, cửa sổ 5 giây | access pending | Task thứ tư nếu tải và preprocess chạy được đúng hạn |

### VEP quarantine

- Không dùng VEP trong main experiments vì label suy từ recording folder, không có marker/annotation để khôi phục stimulus timing, và phase-crossover 4-class ở chance.
- Giữ pluggable adapter, manifest và legacy subject-disjoint result để audit mâu thuẫn giữa kết quả cũ và phase-crossover; audit này thuộc paper cũ, không chặn paper hiện tại.

### Shared preprocessing and channel policy

- Patch dài 1 giây; Sleep giữ epoch 30 giây dưới dạng chuỗi patch.
- Common analysis bandwidth: 0,5–40 Hz; resample về 200 Hz.
- Normalization theo code CBraMod: đổi sang microvolt rồi chia 100; không fit z-score hay bất kỳ thống kê train/test nào.
- Dùng channel registry và canonical order cố định. Không dùng learned Conv1D để ánh xạ montage.
- Verify CBraMod downstream code nhận số kênh thay đổi. Nếu hard-code, dùng pad-and-mask vào registry chung.
- PhysioNet main dùng tập 22 kênh tương ứng BCI IV-2a; bản 64 kênh chỉ là robustness optional.
- Sleep-EDF bipolar không được chiếu thành vị trí điện cực giả. Cross-task channel-space analysis giới hạn ở montage có ý nghĩa; Sleep chỉ so trong frequency space.
- TUEV giữ đúng 16 TCP bipolar derivation từ preprocessing CBraMod đã pin; không tự viết lại pipeline 23-to-16.
- TUEV có một reference reproduction giữ nguyên filter upstream 0,3–75 Hz + notch 60 Hz. Main harmonized giữ nguyên montage/event/split code nhưng đổi filter trên raw liên tục thành 0,5–40 Hz trước khi cắt event; không refilter cửa sổ 5 giây.
- Với cửa sổ 1 giây, tránh diễn giải mạnh năng lượng rất thấp dưới khoảng 2 Hz.

## Model And Training Design

- Shared CBraMod backbone, head riêng cho từng task.
- Khi chuyển task, head cũ được đóng băng; backbone vẫn là phần có thể bị ghi đè.
- Không dùng task-specific adapter trong main experiment.
- Sweep nhanh unfreeze `1/2/4/8` block cuối bằng sequential FT trên một order và một seed.
- Đồng thời chạy single-task baselines cho mọi task có trong benchmark đã khóa sau deadline TUEV. Chọn một depth duy nhất: depth nhỏ nhất đạt ít nhất 95% best mean normalized validation BA qua các task đó. Không chọn depth dựa trên mức forgetting.
- Khóa common preprocessing, optimizer family, schedule và validation budget sau pilot.
- Tách single-task run thành (a) converged/early-stopped cho Week-1 gate và trần `F_rel`, và (b) budget-matched đúng 2.500 optimizer step cho Fisher/overlap và đối chiếu CL.
- Log validation curve BCI IV-2a mỗi 100 step tới 2.500 để biết checkpoint budget-matched nằm trước hay sau peak.
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
- Quarantine VEP theo kết quả marker/annotation/phase-crossover audit; audit riêng kết quả subject-disjoint cũ không chặn main work.
- Xin quyền và tải TUEV v2.0.0; deadline cứng 2026-08-19, sau đó tiếp tục với ba task và hạ claim nếu chưa chạy được.
- Verify CBraMod checkpoint, variable-channel behavior, patching và preprocessing.
- Chạy linear probe và single-task FT baselines.
- Chạy sweep FT 1/2/4/8 block nhanh; chọn một depth chính bằng validation rule.
- Tạo Fisher-overlap signatures và kiểm tra same-task reproducibility.
- Chạy sequential FT smoke test end-to-end.

Gate cuối Tuần 1:

- Performance hợp lý so với reference có cùng protocol, không dùng một ngưỡng công bố sai protocol.
- Fine-tune phải tốt hơn linear probe đủ rõ để chứng minh phần plastic thực sự học.
- Fisher overlap phải có độ phân giải: same-task qua seed/split ổn định và tách được cross-task.
- Chạy một PhysioNet single-task reproduction với split CBraMod gốc 70/19/20 không lọc subject để kiểm tra chuỗi preprocessing-to-checkpoint từ bên ngoài.

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
- Nếu RQ2 yếu/null: báo null có giới hạn trong benchmark đã khóa; giữ causal intervention làm headline nếu nó thành công.
- Nếu cả association và intervention đều fail: không claim predictor; đánh giá chuyển journal hoặc reframing trước khi nộp.

### Submission buffer — 2026-09-10 to 2026-09-16

- Sửa paper, kiểm tra reproducibility, references, ICASSP format và supplementary repository.
- Freeze số liệu/hình trước deadline ít nhất 48 giờ nếu có thể.

## Risks And Recovery

- **Novelty overlap với Qian:** cite trực tiếp; novelty dựa vào EEG grounding + systematic sequential benchmark + matched causal intervention.
- **TUEV access/dung lượng:** deadline cứng 2026-08-19; nếu chưa có snapshot chạy được thì khóa benchmark ba task và hạ claim, không để dataset thứ tư làm trôi lịch.
- **VEP recording-level confound:** đã quarantine khỏi main; audit legacy subject-disjoint result là workstream riêng.
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
- [x] Chốt ba dataset chính và TUEV là task thứ tư có điều kiện; VEP bị quarantine.
- [x] Chốt main methods, metrics, memory accounting và page budget.
- [x] Đọc và định vị novelty so với Qian et al. và EvoBrain.
- [x] Audit VEP marker/annotation/phase-crossover và chốt quarantine.
- [x] Khóa exact BCI IV-2a, PhysioNet-MI, Sleep-EDF và shared preprocessing/training-budget protocol.
- [x] Hoàn thành package config/registry/raw-loader/TUEV-adapter/manifest tooling và focused synthetic tests.
- [x] Tải, checksum và source-audit ba dataset công khai đã khóa.
- [x] Tạo immutable dataset/split manifests.
- [x] Nối raw loaders với frozen manifests và pass real-data preprocessing smoke test cho ba task.
- [x] Pin CBraMod code/checkpoint, xác nhận variable shape và pass real-batch GPU forward/loss/backward cho ba task.
- [x] Materialize cache train/validation resumable từ manifest v3 cho cả ba task.
- [x] Khóa classifier family `flatten_mlp` và định nghĩa fair all-patch linear probe.
- [x] Hoàn tất depth-v3 sweep và khóa final-4-block plasticity depth.
- [x] Chạy exact-2.500-step single-task baseline depth 4 và lưu riêng best/final checkpoint cho ba task.
- [x] Hoàn tất one-off PhysioNet upstream reproduction 64-channel/50-epoch.
- [ ] Chạy converged single-task baseline với early stopping đã khóa trên ba task (đang chạy trong `eeg-conv-bci`, `eeg-conv-phys`, `eeg-conv-sleep`; log dưới `results/single_task/converged_v1/`).
- [ ] Hoàn thành Week 1 gate.
- [ ] Hoàn thành Week 2 CL matrix.
- [ ] Hoàn thành Week 3 diagnostic/intervention gate.
- [ ] Hoàn thành Week 4 analysis/full draft gate.
- [ ] Hoàn thành submission package và validation.

## Decisions

- 2026-08-13: Bài là diagnostic/XAI paper, không đề xuất continual-learning method mới.
- 2026-08-13: CBraMod là backbone duy nhất; shared backbone + task-specific heads; không adapter trong main.
- 2026-08-13: Unfreeze-depth sweep 1/2/4/8 là pilot; main dùng một depth chọn bằng single-task validation, không chọn theo forgetting.
- 2026-08-13: BCI IV-2a, PhysioNet-MI và Sleep-EDF là ba task đã khóa; TUEV thay VEP nếu có dữ liệu chạy được trước deadline Tuần 1.
- 2026-08-13: VEP bị quarantine vì không có marker/annotation và phase-crossover ở chance; không dùng để chặn implementation ba dataset còn lại.
- 2026-08-13: BCI IV-2a đọc nhãn E từ `A0xE.mat`, dùng `[2,6)` giây, bỏ EOG/artifact, gộp T+E trong subject và fail loud nếu session không đủ 6 run/288 trial/4 class cân bằng trước artifact.
- 2026-08-13: Real GDF audit xác nhận marker 32766 cũng bao quanh EOG calibration; MI-run audit chỉ đếm block có trial marker và yêu cầu đúng 6 x 48 trial, nhờ vậy A04T short-EOG không bị loại nhầm.
- 2026-08-13: PhysioNet main loại S088/S092/S100/S104, giữ ID gốc và split 70/18/17; chạy đúng một reproduction unfiltered 70/19/20 theo CBraMod.
- 2026-08-13: Sleep Cassette split tuyệt đối theo subject 48/15/15, giữ hai đêm cùng split, gộp N3+N4, bỏ movement/unknown và crop wake theo thời gian ±30 phút.
- 2026-08-13: Freeze manifest v1: BCI main A01–A05/A06–A07/A08–A09; Sleep stratify `age-band x sex` với seed 20260813; PhysioNet giữ cả clean split và reproduction split trong cùng subject manifest.
- 2026-08-13: Real BCI smoke phát hiện MNE xuất 17 channel label trống thành `EEG-0`…`EEG-16`; registry map chúng theo thứ tự montage chính thức. Manifest v3 supersede v1/v2 và freeze thêm `channels.yaml` cùng `common.yaml`.
- 2026-08-13: Mọi training data access đi qua `FrozenManifestSet` + `ManifestEEGLoader`; runtime verify manifest/config digest, source size, split và sample shape. Full raw SHA-256 được audit riêng trước run.
- 2026-08-13: Pin CBraMod code `b9e961...` và official checkpoint revision `500543c...`/SHA-256 `0792cb8...`; strict weights-only load và upstream numerical parity là pre-run requirements.
- 2026-08-13: Backbone xử lý trực tiếp 2/16/22 channel và 4/5/30 patch, không cần pad-and-mask. Main head khóa theo upstream all-patch MLP family; fair linear probe là flatten toàn bộ channel-patch feature rồi một linear layer. Mean-pool linear chỉ còn là negative control.
- 2026-08-13: Depth sweep v2 bị invalid vì `model.train()` bật dropout trong các block đã freeze và làm representation upstream stochastic. Sweep v3 giữ frozen blocks ở eval mode, chỉ classifier và final-N plastic blocks ở train mode.
- 2026-08-13: Cache v3 materialize 2.678/982 BCI trial, 6.300/1.620 PhysioNet trial và 118.662/39.580 Sleep epoch cho train/validation; mọi index bind vào manifest-set digest.
- 2026-08-13: Depth v3 chọn 4 block cuối theo rule đã khóa: mean normalized validation BA 0,4335 vượt ngưỡng 0,4249 (=95% best 0,4473 ở depth 8). Fair all-patch linear probe lần lượt đạt 0,4902/0,4319/0,5802 BA; selected depth đạt 0,5559/0,4711/0,6784 trên BCI/PhysioNet/Sleep.
- 2026-08-13: Budget-matched final step 2.500 đạt subject BA 0,5214/0,4441/0,6717 trên BCI/PhysioNet/Sleep; best validation tương ứng 0,5559@300, 0,4711@300 và 0,6784@2100. `final.pt` là authority cho Fisher/CL, không dùng `best.pt` thay thế.
- 2026-08-13: Audit code upstream xác nhận PhysioNet external reproduction phải dùng 64 channel + CAR + high-pass 0,3 Hz + notch 60 Hz + 50 epoch full-backbone, không phải main harmonized 22-channel. Cache reproduction được tách tên/root để không thể dùng nhầm.
- 2026-08-13: One-seed upstream PhysioNet reproduction (`seed=3407`, PyTorch 2.13.0+cu130) chọn epoch 33 theo validation kappa và đạt test BA 0,6229, kappa 0,4972, weighted-F1 0,6241. Các số này nằm trong 3 SD của mean 5-seed công bố 0,6417±0,0091 / 0,5222±0,0169 / 0,6427±0,0100; đây là external pipeline gate, không thay thế reproduction đủ 5 seed và không đi vào main result.
- 2026-08-13: Converged baseline khóa validation mỗi 100 step, patience 10 validation, `min_delta=0`, restore-best và hard cap 5.000 step. Patience 5 bị loại vì retrospective simulation trên curve đã khóa sẽ dừng Sleep ở step 1.600 trước peak quan sát tại step 2.100.
- 2026-08-13: Shared preprocessing là 200 Hz, 0,5–40 Hz và microvolt/100; budget-matched là 2.500 step nhưng converged baseline được chạy riêng.
- 2026-08-13: Audit code TUEV đã pin xác nhận pickle ở µV và loader upstream chia 100; main đổi duy nhất filter raw thành 0,5–40 Hz, còn một reproduction giữ nguyên 0,3–75 Hz + notch 60 Hz.
- 2026-08-13: Main `R` dùng balanced accuracy; pairwise outcome chính là `F_rel`; bỏ FWT.
- 2026-08-13: Replay equalized theo persistent bytes; EWC memory gồm Fisher + `theta*`; DER++ giữ reservoir gốc.
- 2026-08-13: Prospective Fisher overlap không phải novelty độc lập vì Qian et al. đã có logic gần tương đương trong Whisper.
- 2026-08-13: Can thiệp high-overlap vs matched random-freeze tại matched new-task performance là bằng chứng mạnh nhất.
- 2026-08-13: Main paper giới hạn 2 figures + 1 table; channel–frequency chỉ là panel nhỏ để EEG grounding.

## Validation

- Focused proof:
  - Unit tests cho `R`, AA, BWT, AF, `F`, `F_rel`, near-chance handling và memory accounting.
  - Dataset audit: unique subject across split; recording integrity; exact exclusion/event inventories; class/age/sex distribution.
  - Shape/mask smoke tests cho 2/16/22-channel inputs và 30-second Sleep sequence.
  - Deterministic same-task Fisher reproducibility check.
- Integration proof:
  - Một sequential FT run end-to-end qua mọi task trong benchmark đã khóa, gồm checkpoint, `R[i,j,s]`, overlap extraction và intervention mask.
  - Recompute main table/figures từ frozen result artifacts bằng một command documented trong repository.
- Statistical proof:
  - Subject-clustered uncertainty, order/seed handling, leave-one-pair-out sensitivity và raw-`F` sensitivity.
  - Matched-performance/Pareto comparison cho high-overlap versus random-freeze.
- Repository-required checks:
  - Plan progress và decisions được cập nhật sau mỗi gate.
  - Không overwrite manifest/config/result đã dùng trong paper.
  - Final manuscript numbers trace được về immutable run IDs.

Observed implementation proof on 2026-08-13:

- `python3.12 -m py_compile` passed for all source and test modules.
- `UV_CACHE_DIR=/tmp/eeg-forgetting-uv-cache uv run pytest -q`: 62 tests passed, including manifest/config tamper checks, main/reproduction cache round-trip, CBraMod identity/variable-shape/depth/train-mode checks, multiclass reproduction metrics, fixed early-stopping contract, overwrite guards, and a real MNE RawArray filter/resample/channel-order integration test.
- ZIP CRC passed for both BCI archives; PhysioNet and Sleep-EDF downloaded-file SHA-256 checks passed against official `SHA256SUMS.txt`.
- `scripts/audit_downloaded_data.py` observed 18 valid BCI GDF sessions, 109 x 6 PhysioNet imagery runs with explicit clean exclusions, and 153 paired Sleep recordings across 78 subjects.
- `scripts/audit_manifests.py --verify-sources` passed for active immutable manifest v3: BCI 5/2/2 subjects, PhysioNet clean 70/18/17 plus 4 explicit exclusions, and Sleep 48/15/15 subjects with 94/30/29 recordings. Every source, config, and manifest SHA-256 matched.
- Real preprocessing smoke through manifest v3 passed on A01E (281 non-artifact trials, 4 classes, `22x4x200`), S001R04 (15 trials, `22x4x200`), and SC00 night 1 (841 epochs, 5 stages, `2x30x200`); every signal was finite `float32` in CBraMod units.
- CBraMod adapter strict-loaded all 211 checkpoint tensors (4,924,000 parameters) and matched pinned upstream output exactly (`max_abs_diff=0.0`). On one L40S with PyTorch 2.13.0+cu130, real batch-size-2 forward/loss/backward passed for BCI, PhysioNet and Sleep with finite non-zero last-block gradients.
- PhysioNet reproduction cache matched the pinned upstream preprocessing exactly on S001R04 (`15x64x4x200`, `max_abs_diff=0.0`), contained the same 9.837 examples across 70/19/20 subjects, and reproduced the upstream classifier initialization bit-for-bit. The completed 50-epoch run selected epoch 33 and produced test BA 0,6229, kappa 0,4972 and weighted-F1 0,6241; checkpoint/config digests re-verified after the run.

## Result

Data-loader foundation, ba public raw-data snapshot, manifest/cache v3, pretrained CBraMod adapter, fair linear probes, head/depth selection, exact-2.500-step baselines và one-off PhysioNet reproduction đã hoàn thành. Ba converged baselines đang chạy độc lập trong `tmux`; sau khi hoàn tất sẽ tạo Fisher signatures. TUEV vẫn cần xác nhận trước deadline truy cập.
