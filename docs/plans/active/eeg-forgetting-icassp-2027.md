# Execution Plan: Channel–Frequency Explanation Drift in Continual EEG Foundation Models

Date: 2026-08-13
Last updated: 2026-08-18

## Status

Active

## Outcome

Hoàn thành một submission ICASSP 2027 dài 4 trang kỹ thuật (+ trang tài liệu tham khảo nếu cần), kiểm tra câu hỏi:

> Khi CBraMod học tuần tự các EEG task khác paradigm, bằng chứng channel–frequency mà model dùng cho task cũ thay đổi như thế nào, và việc giữ hiệu năng có đồng nghĩa với giữ cách model sử dụng tín hiệu sinh lý hay không?

Bài không đề xuất phương pháp continual-learning mới. Đóng góp dự kiến:

1. Xây dựng channel–frequency reliance map bằng held-out occlusion trên cùng subject/mẫu trước và sau mỗi lần chuyển task.
2. Định nghĩa và đo physiological explanation drift (PED), rồi đối chiếu PED với relative forgetting mà không coi accuracy là đại diện cho explanation stability.
3. Kiểm tra liệu Sequential FT, EWC và DER++ có bảo tồn explanation tương ứng với mức bảo tồn hiệu năng hay không; joint training chỉ là offline reference.
4. Kiểm chứng độ tin cậy của attribution bằng random-occlusion fidelity, split/bootstrap null và Integrated Gradients rank agreement trước khi scale.

Deadline mục tiêu: 2026-09-16.

## Context

- [Plan ban đầu](../../../plan%20new%20paper.md) là nguồn ý tưởng; file này thay thế nó làm execution plan hiện hành.
- CL infrastructure đã hoàn tất: CBraMod, ba dataset, ba order, ba seed, Sequential FT/EWC/DER++, joint reference, BCI split robustness, immutable checkpoints và subject-level predictions.
- Fisher instrument đo ổn định nhưng không ủng hộ hypothesis `overlap cao -> quên nhiều`; directional gradient/drift không cải thiện khả năng giải thích. High-overlap freeze thắng random nhưng final `old_only` control không cho thấy shared overlap thêm lợi ích nhất quán ngoài Fisher của task cũ. Toàn bộ nhánh này được đóng băng làm secondary negative audit, không còn là headline.
- CLEX đã nghiên cứu explanation drift trong continual learning tổng quát; novelty ở đây không phải khái niệm drift riêng lẻ mà là drift có cấu trúc sinh lý channel–frequency trong cross-paradigm EEG foundation-model adaptation: <https://www.sciencedirect.com/science/article/pii/S0925231224007318>.
- XAI tĩnh trên EEG foundation models đã có, gồm AttnLRP và causal feature audit; bài không claim “XAI cho EEG-FM đầu tiên”: <https://arxiv.org/abs/2605.17562>, <https://arxiv.org/abs/2605.11410>.
- EvoBrain là phương pháp continual EEG-FM và phải được nêu đích danh; bài này nằm ở lớp lifetime explainability, không reimplement EvoBrain và không đề xuất cơ chế CL mới: <https://arxiv.org/abs/2606.01767>.
- Audit trực tiếp VEP cho thấy 0 marker trong 64 JSON, 0 annotation trong 63 EDF, và phase-crossover BA xấp xỉ chance 0,25. VEP bị quarantine khỏi RQ2, CL matrix và physiological claims; manifest/adapter cũ vẫn được giữ để audit paper trước.
- Ba dataset công khai đã được materialize dưới `data/raw/`: BCI IV-2a có 18 GDF + official true labels, PhysioNet-MI có 109 x 6 imagery-run EDF, Sleep Cassette có 153 cặp PSG/hypnogram trên 78 subject. Checksum/ZIP CRC và source-level annotation audit đều pass.
- Quy định ICASSP: 4 trang nội dung kỹ thuật, trang thứ năm chỉ dành cho references: <https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/>.

## Research Questions And Claims

### RQ1 — Explanation drift

Sau khi học task mới `j`, channel–frequency reliance map của task cũ `i` thay đổi bao nhiêu so với checkpoint ngay trước `j` và offline joint reference?

### RQ2 — Performance–explanation alignment

Việc EWC/DER++ giảm performance forgetting có đồng thời giảm explanation drift so với Sequential FT không, hay accuracy retention và explanation retention tách rời?

Hai outcome đều có giá trị nếu attribution gates pass:

- PED tăng cùng `F_rel`: forgetting về hiệu năng đi cùng thay đổi bằng chứng sinh lý.
- PED vẫn lớn khi `F_rel` nhỏ: accuracy đơn thuần không đủ đánh giá continual EEG model.

Không claim attribution là biological engram, không claim PED là nguyên nhân của forgetting, không claim universal relation, và không claim “first” nếu chưa có systematic literature review đầy đủ.

### Positioning trong Intro

Thông điệp delta dự kiến:

> Prior work proposes methods for continual EEG foundation-model adaptation and separately studies static EEG-FM interpretability. We instead ask whether the physiologically structured evidence behind old-task predictions remains stable throughout continual adaptation.

EvoBrain phải được nêu riêng: họ đề xuất cơ chế continual EEG; bài này phân tích explanation lifetime của các CL baseline đã khóa và không đề xuất cơ chế CL mới.

## Scope

In scope:

- Một backbone duy nhất: CBraMod.
- Ba task đã khóa: BCI Competition IV-2a, PhysioNet-MI và Sleep-EDF Expanded.
- Tái dùng immutable checkpoints của Sequential FT, EWC, DER++ qua ba task order và ba seed; không train lại main CL matrix.
- Joint training là offline explanation reference, không phải continual method.
- Pilot bắt buộc: direct BCI→Sleep, seed 3407, single-task BCI trước transition, Sequential sau Sleep và joint reference.
- Occlusion channel×frequency là XAI chính; Integrated Gradients trên spectral mask coefficients chỉ là kiểm tra rank agreement.
- Reliance/drift luôn so trong cùng task và cùng montage; không so trực tiếp spatial cell giữa Sleep và MI.
- Hai hình và một bảng trong main paper.

Out of scope:

- LaBraM và backbone thứ hai.
- Đề xuất CL algorithm mới.
- Adapter riêng theo task trong main experiment.
- Learned Conv1D channel mapping.
- Bất kỳ model retraining hoặc CL method mới.
- TUEV, VEP, EvoBrain reimplementation và dataset thứ tư.
- Fisher overlap, directional gradient/drift và parameter-freeze intervention trong main claim; chúng chỉ còn là frozen secondary audit/repository artifact.
- Grad-CAM, full per-class attribution atlas, PhysioNet 64-channel và attribution-guided mitigation.
- Method shopping nếu IG/occlusion gate fail.
- Cohort split giả lập thành nhiều task.
- Dùng chữ “engram” cho post-hoc attribution map.

## Dataset Protocol

| Task | Main definition | Subject | Vai trò |
|---|---|---:|---|
| BCI IV-2a | 4-class motor imagery, 22 kênh | 9 | MI low-subject; cần split robustness riêng |
| PhysioNet-MI | Runs 04/06/08/10/12/14, bỏ T0, 4-class theo run; main dùng 22 kênh khớp BCI IV-2a; clean split giữ 105 subject | 105 main / 109 reproduction | Cặp cùng paradigm, khác dataset |
| Sleep-EDF Expanded | 5-class sleep staging, 30 giây, bipolar | 78 | Task khác paradigm; reliance/drift chỉ so trong montage Sleep |

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
- Sleep-EDF bipolar không được chiếu thành vị trí điện cực giả. PED luôn là before/after comparison trong cùng task nên không cần giả lập spatial correspondence giữa Sleep và MI.
- Với cửa sổ 1 giây, tránh diễn giải mạnh năng lượng rất thấp dưới khoảng 2 Hz.

## Model And Training Design

- Shared CBraMod backbone, head riêng cho từng task.
- Khi chuyển task, head cũ được đóng băng; backbone vẫn là phần có thể bị ghi đè.
- Không dùng task-specific adapter trong main experiment.
- Sweep nhanh unfreeze `1/2/4/8` block cuối bằng sequential FT trên một order và một seed.
- Single-task baselines và depth selection đã khóa. Main dùng depth nhỏ nhất đạt ít nhất 95% best mean normalized validation BA qua ba task, không chọn depth dựa trên forgetting hoặc attribution.
- Khóa common preprocessing, optimizer family, schedule và validation budget sau pilot.
- Tách single-task run thành (a) converged/early-stopped cho external performance gate và (b) budget-matched đúng 2.500 optimizer step làm source/reference cho CL trajectory.
- Log validation curve BCI IV-2a mỗi 100 step tới 2.500 để biết checkpoint budget-matched nằm trước hay sau peak.
- Method-specific hyperparameter được tune với cùng validation budget; không tune lại theo order/seed.

Main methods:

1. Sequential FT: baseline gây quên.
2. EWC: regularization baseline; memory accounting gồm Fisher và `theta*` cho toàn bộ tham số plastic.
3. DER++: replay baseline, giữ reservoir policy gốc.
4. Joint training: offline performance/explanation reference, không đưa vào diễn giải như continual method.

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

### Channel–frequency reliance map

- Năm band cố định, dùng half-open interval trừ band cuối: delta `[0.5,4)`, theta `[4,8)`, alpha `[8,13)`, beta `[13,30)`, gamma `[30,40]` Hz.
- Mỗi cached sample được nối các patch 1 giây về đúng continuous trial/epoch của nó trước rFFT, rồi reshape lại nguyên dạng sau inverse-rFFT; không FFT riêng từng patch.
- Một cell là một cặp `(channel, band)`. Occlusion chính đặt hệ số rFFT của band đó về zero chỉ trên channel đó rồi inverse-rFFT; mọi channel/band khác giữ nguyên. Không refilter và không fit baseline từ test data.
- Cell reliance của subject là `BA_unoccluded - BA_cell_occluded`; aggregation giữa subject là mean subject-level drop, không pool trial.
- Validation sample IDs được khóa thành hai nửa class-stratified trong từng subject: `attribution_fit` để xếp hạng cell và `attribution_gate` để kiểm fidelity/PED null. Test subjects không được đọc trong pilot và chỉ dùng báo kết quả cuối sau khi toàn bộ protocol đã khóa.
- Fidelity control: occlude top 20% cell xếp hạng trên `attribution_fit`, đánh giá trên `attribution_gate` và so với 100 random masks cùng số cell. Gate pilot yêu cầu subject-balanced BA drop của top mask lớn hơn percentile 95 của random controls.
- Integrated Gradients được tính theo các hệ số spectral mask differentiable, từ all-zero tới all-one, cho true-class logit; positive attribution được aggregate theo subject rồi xếp hạng cùng cell registry.
- IG là validation instrument, không phải contribution độc lập. Gate pilot yêu cầu Spearman rank agreement với occlusion ít nhất `0,30`.

### Physiological explanation drift

- Với old task `i` ngay trước và sau khi học new task `j`, đánh giá đúng cùng frozen subject/sample set.
- Từ positive part của subject-level reliance map, L1-normalize thành phân bố `P_before` và `P_after`; fail loud nếu map không có positive mass.
- Primary PED là Jensen–Shannon divergence `JSD(P_before, P_after)`. Secondary sensitivity là `1 - Spearman(rank_before, rank_after)`.
- Offline alignment là `JSD(P_after, P_joint)` với joint checkpoint cùng seed và task.
- Instrument-noise null được tạo bằng repeated split/bootstrap của cùng checkpoint và cùng subject. Gate pilot yêu cầu observed before/after PED lớn hơn percentile 95 của null nếu muốn claim explanation changed.
- Mọi inference dùng subject, seed, order và directed transition làm đơn vị phù hợp; channel×frequency cell không phải independent sample.

### Completed Fisher audit

- Fisher split-half/cross-task instrument, overlap–forgetting, directional audit và high/random/old-only interventions được giữ immutable để audit/repository.
- Main paper tối đa dùng một câu hoặc một small secondary statistic để giải thích lý do pivot; không dùng Fisher làm headline figure/RQ và không mở thêm experiment branch.

## Analysis Design

- Ba task tạo sáu directed transition. Tái dùng ba order và ba seed đã khóa; không bổ sung order sau khi thấy PED.
- Mỗi observation chính ghép `(method, order, seed, directed transition, old-task subject)` với `F_rel` và PED before/after trên cùng subject.
- Primary visual: PED vs `F_rel`, màu theo Sequential/EWC/DER++, hiển thị transition và uncertainty theo subject/seed/order.
- Primary method comparison ghép cùng order/seed/transition để hỏi phương pháp giảm `F_rel` có đồng thời giảm PED không.
- Joint reference chỉ dùng đo offline alignment, không trộn vào continual-method comparison.
- Báo raw `F`, `F_rel`, PED, rank-drift sensitivity và BCI split robustness. Không dùng số cell/trial để giả tăng sample size.
- Case “accuracy retained nhưng PED lớn” phải được predefine bằng `|F_rel| <= 0,05` và PED vượt percentile 95 null; không chọn case chỉ vì heatmap đẹp.
- Kết luận phải ổn định về hướng qua ít nhất hai task order và không phụ thuộc duy nhất một seed.

## Attribution Validation And Scale Gate

### Reliability amendment v2 — 2026-08-17

XAI v1 dùng single-cell BA drop không đạt điều kiện đo lường: BCI Before
split-half signed-map cosine chỉ khoảng `0,23` trên 18 main cells và 16/18
BCI←Sleep PED không vượt same-checkpoint null. Các raw PED cũ được giữ làm
diagnostic failure, không còn là bằng chứng RQ2.

Trước mọi XAI scale mới, chạy đúng một reliability pilot trên BCI-only Before
checkpoint seed 42, cùng frozen validation fit/gate và không đọc test. So sánh:

1. single-cell BA-drop v1 đã lưu;
2. single-cell classification-margin drop;
3. 200 randomized masks, mỗi mask che 20% cell, rồi fixed-ridge regression từ
   margin drop sang 110 cell coefficients.

Primary go/no-go là split-half cosine của ridge coefficient map: `>=0,70` pass,
`0,40–0,70` chỉ cho phép một lần tăng sample/mask budget, `<0,40` dừng XAI.
Held-out mask prediction phải có finite positive `R²`. Full three-task/method
scale bị cấm trước khi pilot này pass. Nếu pass, drift v2 dùng symmetric
cross-minus-within estimator; repeated splits chỉ đo uncertainty, không được
coi là independent inferential samples.

Pilot outcome loại ridge vì held-out `R²` âm dù coefficient cosine cao. Protocol
candidate được rút gọn thành single-cell margin-drop. Reliability unit chính là
subject: BCI A06/A07 đạt `0,9226/0,8040`; group-mean chỉ là secondary. Gate kế
tiếp dùng đúng một PhysioNet-Before-Sleep seed-42 checkpoint: mọi subject cosine
phải finite, median `>=0,70`, và ít nhất 2/3 subject `>=0,50`. Không PED/IG/full
scale trước gate này.

### Sleep sample-size amendment v3 — 2026-08-17

PhysioNet gate fail có thể do số trial mỗi subject, không do estimator. Bằng chứng:
cắt BCI margin-drop xuống đúng ngân sách PhysioNet (44 row/subject/nửa) cho cosine
`0,722/0,541`, cũng dưới gate. Sleep-EDF có median `2.386` epoch/subject nên là task
duy nhất mà ràng buộc này không bị dữ liệu chặn; cap 20 hiện tại tự giới hạn xuống
48 row/subject/nửa, gần y hệt PhysioNet. Chạy đúng một pilot để phân biệt hai
nguyên nhân, rồi mới quyết định đóng XAI vĩnh viễn.

Đối tượng: Sleep-Before-PhysioNet, sequential reverse stage 1, seed 42.

- checkpoint `results/continual/sequential_ft_v1/reverse/seed-42/stage-01-sleep_edf_sc/checkpoint.pt`
- sha256 `397d6d60437a55cf5745635c37e276ef99daca8c0dceebb198f0aab54eed7924`
- estimator: single-cell margin-drop; ridge branch vẫn đóng vĩnh viễn
- cell registry: `FPZ-CZ, PZ-OZ` × 5 band = 10 cell
- split: `frozen_stratified_capped_halves`, seed `20260821`, `maximum_rows_per_subject_class: 200`

Chạy **một** lần forward duy nhất ở cap 200, không tạo hai config. Các tập được
chọn là lồng nhau vì sha256 rank được tính trước khi cắt theo cap (verified:
cap 20 ⊂ cap 50 ⊂ cap 200), nên reliability ở mọi mức nhỏ hơn được tính offline từ
cùng margin artifact: 0 GPU thêm, và các điểm là paired thay vì hai run rời.

Curve bắt buộc báo, đơn vị row/subject/nửa: `48` (= cap 20 hiện tại), `121`, `237`, `451`.

Cấm cap 400: class composition lệch rõ (lớp 2 lên 27%, lớp 3 xuống 11%, so với
20%/18% ở cap 20) nên reliability tăng sẽ lẫn confound thành phần lớp. Ở cap 200 chỉ
lớp 3 tụt `18% -> 14%`, phần còn lại gần như không đổi.

Chi phí: `13.559` row × 11 mask ≈ 149k forward pass, khoảng một nửa pilot BCI v2.

Gate giữ nguyên như PhysioNet để hai task so sánh được: unit là subject, mọi cosine
phải finite, median `>=0,70`, và ít nhất 2/3 subject `>=0,50`. Group cosine chỉ là
secondary và không override subject gate. Để cap lớn không bị confound bởi lớp
hiếm cạn trước, margin-drop được mean trong từng class trước rồi equal-average
giữa các class hiện diện của subject; không pool row theo class frequency.

Ba outcome khai báo trước khi chạy, cả ba đều kết luận được:

1. Reliability tăng theo `n` và vượt gate: sample size là nguyên nhân, Sleep đo được, mở PED v2 cho Sleep.
2. Phẳng và thấp ở mọi `n`: sample size không phải nguyên nhân, đóng XAI, negative result mạnh hơn vì loại được giả thuyết thiếu dữ liệu.
3. Tăng nhưng plateau dưới gate: Sleep fail kèm trần định lượng, đóng XAI.

Fidelity leg chạy cùng lượt này ở hai chỗ, vì hiện chỉ có reliability mà chưa có
faithfulness cho margin-drop:

- Sleep: 10 cell nên `top_fraction: 0,20` chỉ chọn 2 cell và chỉ tồn tại `C(10,2)=45` tập (log cũ xác nhận code tự cắt xuống 44 control, percentile 95 quá thô). Dùng top `3/10` và **liệt kê đủ** `C(10,3)=120` tập thay vì 100 draw ngẫu nhiên.
- BCI: fidelity của margin-drop còn nợ từ reliability pilot v2; chạy trên đúng BCI-Before seed-42 checkpoint đã khóa.

Không đọc test. Không train lại. Không mở PED/IG/full method scale trước khi gate này pass.

### High-Gamma replacement candidate v4 — 2026-08-17

Người dùng phê duyệt đúng một dataset-replacement gate sau khi PhysioNet được xác
nhận không phù hợp cho subject-level 110-cell attribution. Chưa thay main matrix
và chưa rerun CL. Candidate được chọn trước khi nhìn XAI vì đạt tiêu chí độc lập:
14 subject, khoảng 963 trial/subject, 4 motor classes, 128 kênh, 4 giây/trial.

- source: NEMAR `nm000172` v1.0.2, DOI `10.82901/nemar.nm000172`, CC-BY-4.0;
- download: chỉ BIDS signal+sidecar cần thiết, không tải `sourcedata` duplicate;
- main montage candidate: đúng 22 kênh BCI IV-2a chung, tạo 110 cells;
- preprocessing: 0,5–40 Hz, 200 Hz, microvolt/100, cửa sổ `[0,4)` từ event onset;
- labels: left hand, right hand, feet, rest; fail loud nếu event/class lệch;
- subject split khóa theo ID trước performance: train `1–7`, validation `8–11`,
  test `12–14`; test không đọc trong candidate gate;
- train đúng một single-task CBraMod seed 42, final-4 blocks, 2.500 steps;
- gate trước full replacement: per-subject single-cell margin reliability trên 4
  validation subjects phải finite, median `>=0,70`, ít nhất 2/3 subject `>=0,50`;
  chỉ nếu pass mới chạy margin fidelity. Fail thì không thay PhysioNet và không
  rerun Sequential/EWC/DER++.

High-Gamma có motor execution/rest signal mạnh và có thể chứa movement/EMG
confound; mọi claim sau này phải gọi đúng motor-task dataset, không tự gọi pure MI.

### High-Gamma attribution gate v5 — 2026-08-18

Khóa trước khi chạy XAI trên checkpoint final step 2.500 seed 42. Dùng toàn bộ
3.934 validation row của subject 8–11, chia `sha256_rank_within_subject_class_v1`
với seed `20260818` thành 1.966 fit và 1.968 gate row; không cap vì số trial/subject
là lý do độc lập để chọn candidate. Assignment SHA-256 là
`52add2ae02588b2c68ca3e4b423099e379e9cb27ba762dbf38c11b447696449b`.

- Reliability dùng single-cell classification-margin drop, mean trong từng class
  rồi equal-average giữa class. Gate giữ nguyên: mọi cosine finite, median subject
  `>=0,70`, và ít nhất 2/3 subject `>=0,50` (với 4 subject nghĩa là ít nhất 3/4).
- Chỉ nếu reliability pass, fidelity chọn top `22/110` cell từ fit map và đánh giá
  trên gate half, so với 100 unique random mask cùng 22 cell, seed `20260821`.
- Fidelity pass khi cả subject-balanced BA drop và class-balanced margin drop của
  top mask đều lớn hơn percentile 95 của random controls.
- Chỉ khi reliability và fidelity cùng pass mới cho phép thay PhysioNet và thiết
  kế lại main CL matrix. Test subject 12–14 vẫn cấm đọc trong cả hai gate.

### High-Gamma replacement matrix v6 — 2026-08-18

Vì v5 pass, main replacement matrix được khóa trước khi train CL:

- canonical tasks: `BCI IV-2a`, `High-Gamma`, `Sleep-EDF`; PhysioNet artifacts cũ
  giữ immutable nhưng không thuộc replacement matrix;
- orders: forward `BCI→High-Gamma→Sleep`, reverse
  `Sleep→High-Gamma→BCI`, challenging `High-Gamma→BCI→Sleep`;
- seeds `3407/42/2026`; final-4 blocks, 2.500 step/task, optimizer/schedule và
  Sequential FT/EWC `lambda=100.000`/DER++ `8 MiB` giữ nguyên, không retune;
- chạy đủ Sequential trước; chỉ mở EWC/DER++ sau khi cả 9 Sequential run hợp lệ;
- evaluation dùng frozen test cache. High-Gamma test 12–14 chỉ được materialize
  sau khi v5 result đã immutable; không dùng test để đổi gate/hyperparameter;
- mọi run dùng config/result/checkpoint/prediction digest và resumable stage như
  matrix cũ. PED chỉ scale sau khi performance matrix và checkpoint inventory pass.

### Near-chance stability amendment v7 — 2026-08-18

Người dùng phê duyệt sau khi v6 hard gate fail vì đúng một `Sleep←BCI` replicate
có headroom `0,0482 < 0,05`. Không hạ threshold và không tạo `F_rel` giả. Quy tắc
v7 tạo summary mới, không overwrite v6:

- report table vẫn giữ `F_rel=null` cho replicate invalid và báo raw F;
- riêng stability gate, nếu một directed transition thiếu replicate vì near-chance,
  dùng raw F cho **toàn bộ replicate của direction đó**, không trộn raw và relative;
- fallback khóa duy nhất cho `sleep_edf_sc<-bciciv2a`; mọi direction khác tiếp tục
  dùng `F_rel`;
- giữ nguyên minimum n=3, signal `|mean| > sample SD`, sign fraction `>=2/3`, và
  yêu cầu ít nhất bốn signal/sign-consistent directions;
- original failed summary được giữ immutable. Chỉ khi digest-bound v7 summary pass
  mới mở replacement EWC/DER++.

### Replacement PED scale v8 — 2026-08-18

Khóa trước khi chạy XAI trên replacement checkpoints:

- chỉ old-task BCI và High-Gamma được đo PED vì cả hai đã pass reliability và
  fidelity; old-task Sleep bị loại vì fidelity fail;
- dùng đúng validation split đã khóa: BCI `489/493`, High-Gamma `1966/1968`;
- mỗi checkpoint tạo single-cell class-balanced margin-drop map cho fit/gate,
  theo subject và group, trên cùng 110 cell;
- primary noise-corrected PED cho mỗi subject là
  `0,5[JSD(B_fit,A_gate)+JSD(B_gate,A_fit)] -
   0,5[JSD(B_fit,B_gate)+JSD(A_fit,A_gate)]`;
- không clip PED âm; cross-JSD và within-checkpoint noise được báo riêng;
- scale mọi immediate transition hợp lệ trong 27 Sequential/EWC/DER++ run,
  tổng cộng 63 transition cell. Không chạy lại fidelity per checkpoint và không
  đọc test cho attribution; performance được ghép từ immutable test result đã có;
- joint/offline alignment là bước riêng sau primary before/after PED, không chặn v8.

### Joint offline-alignment hypothesis gate v9 — 2026-08-18

Người dùng phê duyệt một bounded joint hypothesis test; manuscript hiện tại giữ
nguyên cho tới khi gate pass. Joint mới là task-balanced offline reference, không
phải explanation lý tưởng hay ground truth.

- task set `[bciciv2a, high_gamma, sleep_edf_sc]`, seeds `3407/42/2026`;
- giữ nguyên 2.500 update/task, canonical round-robin, optimizer/LR/depth của
  `joint_v1`; không retune;
- mixed `cache_roots` giống replacement CL configs;
- chỉ score joint explanation cho BCI và High-Gamma trên frozen validation
  fit/gate; Sleep tiếp tục bị loại;
- strict all-six gate: BCI/High-Gamma × 3 seed đều phải pass reliability
  (finite, median subject cosine `>=0,70`, `>=2/3` subject `>=0,50`) và held-out
  fidelity về cả margin drop lẫn subject-balanced BA drop;
- nếu một map fail, joint analysis dừng và không vào paper;
- nếu pass, seed-matched alignment dùng cùng noise-corrected symmetric JSD:
  `A_before=A(Before,Joint)`, `A_after=A(After,Joint)`,
  `delta_A=A_after-A_before`; cross-seed variation chỉ là robustness, không phải
  null;
- hypothesis chỉ pass nếu EWC hoặc DER++ có paired mean `delta_A` thấp hơn
  Sequential, 95% order-seed block-bootstrap CI không chứa 0, và thắng ít nhất
  8/9 matched order-seed blocks;
- nếu pass, paper chỉ được thêm tối đa một aggregate `delta_A` statistic/method
  và 2–3 câu; direction/seed detail xuống supplementary. Nếu fail/null, bỏ joint
  khỏi manuscript và không mở thêm experiment.

### Post-gate exploratory joint seed audit v10 — 2026-08-18

Sau formal v9 fail, người dùng yêu cầu xem thêm seed. Chạy đúng ba seed cố định
`[7,123,999]` với cùng training và XAI gates; không chọn seed theo outcome, không
retune và không tính `delta_A`. Đây là sensitivity audit post hoc: chỉ báo pass
rate và distribution, không được lật formal v9 decision hoặc đi vào main paper.

Pilot duy nhất trước scale là direct `bciciv2a -> sleep_edf_sc`, seed `3407`:

1. Tạo BCI reliance map tại exact BCI-only source checkpoint.
2. Tạo cùng map tại Sequential checkpoint sau khi học Sleep.
3. Tạo BCI map từ joint seed-3407 làm offline reference.
4. Chạy held-out top-20%-vs-random fidelity gate.
5. Chạy spectral-mask IG và rank-agreement gate.
6. Ước lượng same-checkpoint bootstrap/split null rồi so observed PED.

Scale sang toàn bộ checkpoint hiện có chỉ khi cả ba gate pass: fidelity, IG agreement và observed PED vượt same-checkpoint null. PED không vượt null không làm attribution implementation sai, nhưng cấm claim explanation changed; khi đó dừng scale và báo pilot null. Mức performance forgetting chỉ chọn framing “coupled” hoặc “decoupled”, không được dùng để thay đổi attribution protocol.

## Paper Budget

- Intro: khoảng 0,6 trang.
- Related work: khoảng 0,3 trang.
- Method: khoảng 0,9 trang.
- Experimental setup: khoảng 0,6 trang.
- Results: khoảng 1,3 trang.
- Conclusion: khoảng 0,2 trang.

Main artifacts:

- Figure 1: protocol cùng reliance map before/after/delta của transition tiêu biểu được chọn bằng rule, không chọn bằng thẩm mỹ.
- Figure 2: PED vs `F_rel` và paired method comparison; đây là headline figure.
- Table 1: subject-aggregated performance forgetting, PED và offline alignment của Sequential/EWC/DER++.

Không thêm hình thứ ba vào main paper.

## Approach And Schedule

### Foundation complete — 2026-08-13 to 2026-08-15

- Dataset/manifests/cache, CBraMod validation, depth/head selection, single-task references và full Sequential/EWC/DER++/joint matrices đã hoàn tất.
- Fisher/directional/intervention branch đã trả lời hypothesis cũ và được frozen làm secondary audit.

### XAI protocol and pilot — 2026-08-15 to 2026-08-18

- Khóa config, spectral occlusion operator, cell registry, subject aggregation, PED và output schema.
- Implement synthetic FFT reconstruction/one-cell removal tests, checkpoint identity verification và immutable result writer.
- Chạy pilot BCI→Sleep seed 3407 trên BCI-only, post-Sleep Sequential và joint reference.
- Chạy fidelity, IG agreement và bootstrap/split-null gates; không đọc test để chỉnh threshold/operator.

Gate pilot:

- Top-20% `attribution_gate` BA drop vượt percentile 95 của 100 random masks.
- Occlusion–IG Spearman ít nhất `0,30`.
- Observed before/after PED vượt percentile 95 của same-checkpoint bootstrap/split null.
- Kết quả finite, deterministic và subject-level aggregation/mask identity được digest-bind.

### XAI scale — 2026-08-19 to 2026-08-23

- Nếu pilot pass, chạy full occlusion trên Sequential/EWC/DER++ qua ba order, ba seed và sáu directed transition từ checkpoint hiện có.
- Chạy IG validation trên một representative transition cho mỗi old task qua ba seed; selection rule phải khóa trước khi mở kết quả.
- Tạo joint-reference map cho ba task/ba seed và PED/offline-alignment summary có digest verification.

### Analysis and figures — 2026-08-24 to 2026-08-28

- Ghép subject-level `F_rel`, PED, method/order/seed; chạy paired comparison, raw-`F`, rank-drift và BCI split sensitivity.
- Chốt framing coupled hoặc decoupled theo rule đã khóa.
- Hoàn thiện hai figures + một table và draft Method/Setup.

### Full draft and review — 2026-08-29 to 2026-09-09

- Viết full four-page draft; nêu CLEX, static EEG-FM XAI và EvoBrain delta trực tiếp.
- Internal review tập trung attribution validity, claim strength, leakage và page budget.
- Nếu pilot attribution fail sau một technical correction, dừng XAI scale và quyết định submit Fisher negative audit hay không; không method-shop để cứu deadline.

### Submission buffer — 2026-09-10 to 2026-09-16

- Sửa paper, kiểm tra reproducibility, references, ICASSP format và supplementary repository.
- Freeze số liệu/hình trước deadline ít nhất 48 giờ nếu có thể.

## Risks And Recovery

- **Static EEG-FM XAI đã đông:** không bán channel–frequency heatmap riêng lẻ; novelty phải nằm ở explanation lifetime/drift trong cross-task continual adaptation.
- **Explanation drift đã có ở domain khác:** cite CLEX trực tiếp; delta là physiological cell structure, EEG-FM, heterogeneous task transition và performance–explanation alignment.
- **Attribution không faithful:** pilot held-out top-vs-random là hard gate; fail sau một technical correction thì dừng scale, không đổi explainer sau khi nhìn kết quả.
- **IG baseline/gradient bất ổn:** IG chỉ là validation. Non-finite hoặc implementation-invalid được sửa một lần bằng cùng predeclared spectral-mask formulation; scientific disagreement với occlusion là gate fail, không phải lý do method-shop.
- **Drift giả do đổi mẫu/class composition:** before/after dùng đúng cùng sample IDs, subject-balanced và class-stratified reporting; không so heatmap tạo từ cohort khác nhau.
- **Montage khác nhau:** chỉ đo drift trong cùng old task; không so trực tiếp cell Sleep với cell MI hay chiếu bipolar thành scalp location giả.
- **VEP recording-level confound:** đã quarantine khỏi main; audit legacy subject-disjoint result là workstream riêng.
- **BCI IV-2a chỉ 9 subject:** per-subject logging và split robustness sớm; không giả vờ tăng power bằng epoch count.
- **Performance forgetting nhỏ:** nếu PED vượt null khi `|F_rel| <= 0,05`, dùng decoupling finding; nếu cả PED và forgetting đều null thì báo null và không chọn transition hậu nghiệm.
- **Page overflow:** giữ đúng 2 figures + 1 table; Fisher audit, per-class atlas, full random distributions và extra montage để repository/supplementary hoặc bỏ.
- **Compute failure:** ưu tiên pilot fidelity -> full occlusion -> PED summary -> selected IG validation; không train thêm model.

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
- [x] Chạy converged single-task baseline với early stopping đã khóa trên ba task.
- [x] Chạy exact per-example Fisher instrument pilot và pass split-half/cross-task gate.
- [x] Chạy sequential FT smoke forward/reverse trên test subjects và lưu `R`/predictions/checkpoints.
- [x] Chạy main sequential FT matrix 3 order × 3 seed và pass stability gate trước EWC/DER++.
- [x] Hoàn thành Week 1 gate.
- [x] Chạy validation pilot BCI→Sleep và khóa EWC `lambda=100.000`, DER++ cap `8 MiB`.
- [x] Chạy main EWC matrix 3 order × 3 seed với offline Fisher state đã khóa.
- [x] Chạy main DER++ matrix 3 order × 3 seed với reservoir 8 MiB đã khóa.
- [x] Tách sáu directed transition và hoàn tất path-free two-task Sequential FT bằng ba chiều còn thiếu × ba seed.
- [x] Chạy BCI robustness fold v2 cho Sequential FT, giữ nguyên mọi hyperparameter.
- [x] Chạy task-balanced joint-training upper bound ba seed.
- [x] Chạy BCI→Sleep high-overlap intervention 1/5/10% với ba matched random masks × ba seed.
- [x] Chạy PhysioNet→BCI localization replication 5/10% × bốn mask × ba seed.
- [x] Chạy directional gradient/drift audit sáu transition × ba seed và tổng hợp descriptive correlation.
- [x] Chạy final old-task-importance control trên BCI→Sleep và PhysioNet→BCI, 5/10% × ba seed (12 run); legacy-schema reader đã sửa và official immutable summary xác nhận gate fail 3/4.
- [x] Chốt pivot từ Fisher-overlap headline sang channel–frequency explanation drift.
- [x] Khóa XAI config/operator/schema và viết synthetic/unit proof.
- [x] Chạy BCI→Sleep attribution pilot ba seed và khóa transparent replication amendment.
- [x] Scale reliability-aware XAI qua replacement checkpoints sau High-Gamma gate pass.
- [x] Chạy reliability amendment v2 trên BCI Before seed 42; full scale vẫn bị chặn.
- [x] Chạy single-cell margin reliability trên PhysioNet Before-Sleep seed 42; gate fail và XAI scale dừng.
- [x] Chạy Sleep sample-size amendment v3: một forward ở cap 200 trên Sleep-Before-PhysioNet seed 42, báo curve `48/121/237/451` row/subject/nửa từ cùng artifact.
- [x] Chạy fidelity leg cho margin-drop: Sleep top `3/10` exhaustive `C(10,3)=120`, và BCI-Before seed 42.
- [x] Đóng XAI: Sleep reliability pass nhưng exhaustive fidelity fail cả margin và BA.
- [x] Audit/download High-Gamma BIDS v1.0.2 và pass one-subject real-data contract.
- [x] Train single-task High-Gamma seed 42 đúng 2.500 step; chưa rerun CL.
- [x] Khóa attribution split và chạy High-Gamma reliability/fidelity gate trên final checkpoint; cả hai pass.
- [x] Chạy High-Gamma replacement Sequential FT matrix 3 order × 3 seed và verify immutable artifacts.
- [x] Pass replacement Sequential stability amendment v7; EWC/DER++ được phép mở.
- [x] Hoàn tất replacement EWC/DER++ 3 order × 3 seed và digest-verified summaries.
- [x] Tổng hợp noise-corrected PED/performance alignment trên 63 transition cell.
- [x] Parameterize joint mixed-task/cache roots và train `joint_high_gamma_v1` ba seed.
- [x] Gate sáu joint maps: strict all-six fail, bỏ joint hypothesis và không tính `delta_A`.
- [x] Chạy exploratory joint seeds `[7,123,999]`; High-Gamma pass 3/3 nhưng không đổi formal gate.
- [x] Hoàn thiện hai figures và một table từ digest-bound replacement summary.
- [x] Viết full ICASSP manuscript source với statistical reporting và limitations.

### Completed execution history — 2026-08-14

- Pairwise, BCI robustness, joint upper bound, BCI→Sleep intervention, PhysioNet→BCI replication và directional v2 đều đã hoàn tất; tên tmux/log cũ bên dưới chỉ là audit trail, không phải active queue.
- BCI robustness fold v2 giữ A03–A07 train, A08–A09 validation, A01–A02 test; không retune.
- Intervention xếp hạng theo geometric mean của Fisher đã L2-normalize trong từng encoder layer. Mỗi tỷ lệ 1/5/10% dùng một high-overlap mask và ba random mask cùng số phần tử ở từng layer, chia sẻ mask qua ba training seed. Sau mỗi AdamW step, phần tử frozen được khôi phục chính xác về anchor để triệt cả decoupled weight decay.
- Follow-up localization replication dùng hướng `physionet_mi → bciciv2a`, tỷ lệ 5/10%, ba matched random mask và ba seed (24 run). Hai tỷ lệ đều phải pass matched-plasticity/protection gate; không dùng lại mức 1% yếu trên validation của cặp đầu.
- Directional audit chạy đủ sáu transition × ba seed tại checkpoint old-task. Mỗi task dùng cố định 1.024 mẫu train để đo signed mean observed-label NLL gradient trên bốn encoder block plastic. Báo negative gradient cosine như conflict score, cùng exact post-training parameter drift được weighted bởi old-task Fisher và geometric-mean shared Fisher.
- Directional audit chỉ mang vai trò descriptive/mechanistic vì có sáu transition; không dùng p-value hoặc claim predictor tổng quát từ sáu điểm này.
- `eeg-replication-gpu0` chạy ratio 5% rồi ba hướng directional đầu trên physical GPU 0; `eeg-replication-gpu2` chạy ratio 10% rồi ba hướng còn lại trên physical GPU 2. `eeg-replication-summary` chờ đủ 24 + 18 result và tạo hai summary có digest verification.
- Recovery 2026-08-14: intervention hoàn tất 24/24. Directional v1 dừng ở 14/18 do checksum của `physionet_to_bci/seed-2026` bị chép sai trong config (`892bcf...`). Giữ nguyên partial artifacts v1 làm audit trail; v2 chỉ sửa đúng digest nguồn, bind rõ v1 bị supersede và chạy lại đủ 18 result, không trộn hai config SHA trong summary.
- Directional audit v2 hoàn tất 18/18 nhưng không giải thích forgetting tốt hơn overlap đối xứng (n=6, descriptive): correlation với mean `F_rel` lần lượt là overlap `-0,815`, gradient conflict `-0,646`, shared-Fisher drift `-0,715`, old-Fisher drift `-0,541`. Không mở thêm nhánh directional.
- Final control chỉ thêm điều kiện `old_only`: freeze top Fisher của task cũ với đúng ngân sách từng layer như high-overlap. Chạy 2 cặp × 2 ratio × 3 seed = 12 run, tái dùng toàn bộ high-overlap result đã khóa. Sau control này dừng experiment branching và chuyển sang chốt analysis/paper, bất kể gate pass hay fail.

### Active execution queue — 2026-08-15

1. Sửa legacy reader và freeze official `old_only` summary; không chạy lại training.
2. Tạo locked XAI pilot config cho direct BCI→Sleep seed 3407, bind exact source/post/joint checkpoint digests, frozen BCI validation `attribution_fit/gate` IDs và untouched test IDs.
3. Implement rFFT cell occlusion, spectral-mask IG, subject-level BA drop, random-mask fidelity và bootstrap/split null với focused tests.
4. Chạy pilot; ghi pass/fail từng gate trước khi tạo full-scale config.
5. Chỉ khi pilot pass, generate immutable full matrix manifest từ checkpoint inventory hiện có và launch inference jobs.

- [x] Hoàn thành Week 2 CL matrix.
- [ ] Hoàn thành XAI attribution pilot gate.
- [ ] Hoàn thành full explanation-drift scale gate.
- [x] Hoàn thành Week 4 analysis/full draft gate; Tectonic PDF/page-count QA pass.
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
- 2026-08-13: Converged runs đạt best subject BA 0,5516@300 cho BCI và 0,4712@300 cho PhysioNet, cùng early-stop ở step 1.300. Sleep đạt 0,6984@4.200 và kết thúc bởi hard cap 5.000 sau tám validation liên tiếp dưới best; checkpoint được giữ nhưng phải ghi là maximum-step capped, không claim đã kích hoạt early convergence.
- 2026-08-13: Fisher instrument v1 dùng observed-label exact per-example empirical Fisher, uniform sample không hoàn lại từ natural train distribution, 1.024 mẫu chia hai half 512 rời nhau, model eval và chỉ final-4 plastic backbone blocks. Primary layer normalization là L2 trong từng encoder layer; top-k sensitivity chưa kích hoạt.
- 2026-08-13: Fisher microbatch giảm 4 xuống 1 sau khi BCI/PhysioNet chạm OOM trên shared GPU có tiến trình ngoài; estimator vẫn là cùng tổng exact per-example squared gradients và sample indices không đổi. Toàn bộ batch-4 attempt được giữ dưới rejected results dù Sleep đã hoàn tất, rồi cả ba task được chạy lại cùng config digest để tránh một instrument set trộn execution config.
- 2026-08-13: Attempt microbatch-1 dùng `torch.func` vẫn tăng VRAM theo thời gian và OOM trên BCI/PhysioNet; thay bằng standard backward từng example, tắt gradient cho mọi tham số ngoài final-4 block. Công thức observed-label per-example Fisher không đổi, nhưng graph được giải phóng mỗi mẫu; toàn bộ attempt cũ tiếp tục được giữ dưới rejected results.
- 2026-08-13: Fisher instrument gate pass: split-half cosine BCI/PhysioNet/Sleep = 0,9957/0,9655/0,9873; cross-task cosine BCI–PhysioNet/BCI–Sleep/PhysioNet–Sleep = 0,8277/0,5672/0,5871. Minimum within-task 0,9655 vượt threshold 0,90 và margin trên maximum cross-task = 0,1378 vượt threshold 0,05. Không cần escalation 2.048 mẫu.
- 2026-08-13: Sequential FT smoke khóa hai order forward `BCI→PhysioNet→Sleep` và reverse để phủ hai chiều mọi cặp ở seed 3407. Mỗi task nhận đúng 2.500 step, optimizer reset, head được preinitialize theo canonical order để không phụ thuộc task order, head cũ freeze, final-4 backbone tiếp tục plastic. Đánh giá trên frozen test subjects sau mỗi stage; `F_rel` chỉ hợp lệ khi before-score cao hơn chance ít nhất 0,05 và không bao giờ clip.
- 2026-08-14: Main sequential FT khóa seeds 3407/42/2026 và order thứ ba challenging `PhysioNet→BCI→Sleep`, tạo ma trận 9 run. Stability gate yêu cầu đủ ít nhất ba replicate hợp lệ cho sáu direction; ít nhất bốn direction phải có `|mean F_rel| > sample SD` và ít nhất 2/3 replicate cùng dấu. Chỉ sau gate này mới scale EWC/DER++.
- 2026-08-14: Cả 9/9 sequential run hoàn tất và stability gate pass: đủ replicate, 6/6 direction vượt signal rule và 6/6 nhất quán dấu. Năm direction có forgetting dương; `BCI←PhysioNet` có backward transfer nhất quán (`mean F_rel=-0,2351`). Ba unordered pair cho Fisher cosine 0,8277/0,5672/0,5871 và mean `F_rel` 0,0193/0,3724/0,3262; Spearman `rho=-1` chỉ là exploratory vì `n=3`, không được diễn giải confirmatory.
- 2026-08-14: Method-selection pilot khóa trước cặp `BCI→Sleep`, seed 3407 và chỉ đọc validation. EWC thử `lambda=0/1.000/10.000/100.000`; DER++ thử cap `0/8/16/32 MiB`, Algorithm R với slot cố định tính đủ signal/label/logit/task metadata, hai replay draw độc lập và `alpha=beta=0,5` theo bài gốc. Candidate hợp lệ phải giữ Sleep validation BA không thấp hơn baseline quá 0,02; trong tập hợp lệ chọn `F_rel` của BCI thấp nhất, hòa thì chọn strength/bytes thấp hơn. Sau selection khóa một giá trị cho mọi order/seed, không tune lại.
- 2026-08-14: Cả 8/8 method-selection candidate hoàn tất và summary bind input bằng SHA-256. Khóa EWC `lambda=100.000` (`F_rel` BCI 0,0096; Sleep val BA 0,6618; 12.883.200 byte mỗi Fisher/anchor state) và DER++ `8 MiB` (`F_rel` 0,0095; Sleep val BA 0,6774; 119 slot/8.381.067 byte allocated). Không mở rộng grid hoặc tune lại theo order/seed.
- 2026-08-14: Full matrices được launch theo cùng 9 cell `forward/reverse/challenging × 3407/42/2026`: `eeg-cl-ewc-main` chỉ thấy GPU vật lý 0, `eeg-cl-derpp-main` chỉ thấy GPU vật lý 2, và `eeg-cl-method-main-summary` chờ đủ 18 immutable `result.json` rồi tự verify/tổng hợp. Mỗi checkpoint lưu model, method state và CPU/CUDA RNG state để stage-boundary resume giữ nguyên trajectory.
- 2026-08-14: EWC/DER++ hoàn tất 18/18 main run và watchdog tạo hai digest-verified summary. Mean pair `F_rel` của Sequential/EWC/DER++ lần lượt là: BCI–PhysioNet `0,0193/-0,0776/-0,0682`; BCI–Sleep `0,3724/0,0726/0,0065`; PhysioNet–Sleep `0,3262/0,0084/0,0662`. EWC inherited signal gate báo false vì chỉ 2/6 direction còn lớn hơn seed noise; đây là forgetting suppression, không phải execution failure. DER++ còn 4/6 và pass. Pair-level Spearman vẫn exploratory với `n=3`: `-1/-1/-0,5`, không ủng hộ positive overlap-risk claim.
- 2026-08-14: Follow-up khóa trước khi chạy: clean directional matrix tái dùng ba first-transition hiện có và chỉ bổ sung `BCI→Sleep`, `Sleep→BCI`, `PhysioNet→Sleep` ở ba seed; BCI robustness fold v2 là train A03–A07/validation A08–A09/test A01–A02 và không retune; joint upper bound dùng canonical round-robin với đúng 2.500 optimizer update/task; intervention BCI→Sleep dùng element-wise geometric-mean Fisher score, freeze 1/5/10% trong từng layer, so với ba random mask cùng layer/count ở ba seed và restore giá trị frozen sau AdamW step.
- 2026-08-13: Shared preprocessing là 200 Hz, 0,5–40 Hz và microvolt/100; budget-matched là 2.500 step nhưng converged baseline được chạy riêng.
- 2026-08-13: Audit code TUEV đã pin xác nhận pickle ở µV và loader upstream chia 100; main đổi duy nhất filter raw thành 0,5–40 Hz, còn một reproduction giữ nguyên 0,3–75 Hz + notch 60 Hz.
- 2026-08-13: Main `R` dùng balanced accuracy; pairwise outcome chính là `F_rel`; bỏ FWT.
- 2026-08-13: Replay equalized theo persistent bytes; EWC memory gồm Fisher + `theta*`; DER++ giữ reservoir gốc.
- 2026-08-13: Prospective Fisher overlap không phải novelty độc lập vì Qian et al. đã có logic gần tương đương trong Whisper.
- 2026-08-13: Can thiệp high-overlap vs matched random-freeze tại matched new-task performance là bằng chứng mạnh nhất.
- 2026-08-13: Main paper giới hạn 2 figures + 1 table; channel–frequency chỉ là panel nhỏ để EEG grounding.
- 2026-08-15: Kết quả Fisher chính bị khóa là negative audit: symmetric overlap không dự đoán positive forgetting risk, directional features không cải thiện, và old-task-only Fisher control làm shared-overlap benefit không còn nhất quán. Không mở thêm Fisher branch.
- 2026-08-15: Headline chuyển sang channel–frequency explanation drift. Quyết định này supersede vai trò “panel nhỏ” ngày 2026-08-13; attribution lifetime giờ là RQ chính, còn Fisher chuyển ra secondary/repository.
- 2026-08-15: Không reimplement EvoBrain, không dùng “engram”, không thêm TUEV/LaBraM/Grad-CAM. Tái dùng toàn bộ CL checkpoints đã khóa.
- 2026-08-15: Pilot authority là direct BCI→Sleep seed 3407. Full scale bị chặn cho tới khi top-vs-random fidelity, spectral-mask IG rank agreement và observed-PED-vs-null gates đều pass.
- 2026-08-15: Occlusion ranking fit và fidelity gate dùng hai nửa class-stratified, frozen trong validation subjects. Test subjects không được đọc trong pilot và chỉ dùng final reporting sau protocol lock. Primary drift là JSD của positive L1-normalized subject maps; cells/trials không phải inference units.
- 2026-08-15: XAI pilot config bind exact BCI-only/post-Sleep/joint checkpoint SHA, validation/test-cache SHA và deterministic validation assignment SHA. Validation A06/A07 được khóa thành 489 `attribution_fit` + 493 `attribution_gate`; pilot có 22 channel × 5 band = 110 cell. Fidelity/IG bắt buộc pass ở before và after; joint được report nhưng không dùng để làm fail gate.
- 2026-08-15: Legacy high-overlap artifacts thiếu `old_task/new_task` chỉ được reader chấp nhận khi config SHA, Fisher task keys và evaluation task keys cùng khớp. Official old-importance summary pass 3/4 final test controls, nên claim gate vẫn fail và kết luận negative audit không đổi.
- 2026-08-15: Pilot launch đầu tiên hoàn tất immutable `before` cell predictions rồi dừng ở fidelity vì NumPy array không có scalar truth value trong empty-index guard. Technical correction duy nhất đổi `not indices` thành `len(indices) == 0`, thêm regression test và không thay đổi config, operator, split, threshold hay artifact đã tính.
- 2026-08-15: Seed-3407 pilot hoàn tất: fidelity pass mạnh ở before/after, PED `0,4286` vượt conservative same-checkpoint null p95 `0,4207`, nhưng hard IG gate cũ fail sát vì after `rho=0,2985 < 0,30` (before `0,3092`, joint `0,5035`). Original result giữ nguyên borderline fail; không round hoặc overwrite.
- 2026-08-15: Trước khi mở seed bổ sung, protocol amendment v2 xác nhận `0,30` là internal heuristic chứ không phải external validity standard. IG chuyển thành continuous supporting evidence; fidelity và PED-vs-null là hard gates. Chạy frozen replication seed 42/2026 và chỉ scale nếu mỗi hard gate pass ít nhất 2/3 seed, đồng thời IG dương ở before/after cho cả ba seed. Không thay attribution operator, sample split, mask, null hoặc checkpoint.
- 2026-08-15: Replication seed 42/2026 hoàn tất. Theo rule per-gate đã nói trước khi chạy: fidelity pass 2/3 seed, PED-vs-null pass 2/3 seed và required-role IG dương 3/3 seed; scale được phép nhưng phải báo instability vì chỉ seed 3407 pass đồng thời cả hai hard gate. Joint IG không phải required-role evidence.
- 2026-08-15: RQ2 scale bắt đầu bằng BCI←Sleep trên main CL matrix: Sequential/EWC/DER++ × order forward/challenging × seed 3407/42/2026 = 18 paired transition cells. Mỗi cell dùng stage-2 before và stage-3 after checkpoint đã được main summary digest-bind, cùng frozen BCI validation split và occlusion/fidelity/PED-null operator; không train lại và không đọc test. IG không lặp trên mọi checkpoint vì đã là supporting instrument kiểm qua ba pilot seed.
- 2026-08-15: BCI←Sleep RQ2 scale hoàn tất 18/18: mean `F_rel/PED` Sequential `0,3016/0,4891`, EWC `0,1087/0,2438`, DER++ `-0,0164/0,3755`. EWC và DER++ cùng giảm cả hai metric ở 5/6 paired order-seed cells; EWC fidelity pass cả role chỉ 3/6 nên explanation-retention claim của EWC yếu hơn DER++ (6/6 fidelity).
- 2026-08-15: Trước khi mở các directed transition khác, khóa task-specific attribution validation cho PhysioNet và Sleep. Cả hai hash-rank tối đa 20 validation row trong từng subject/class rồi chia fit/gate: PhysioNet 720/720, Sleep 732/732. Cap này giữ subject/class coverage, giảm Sleep từ 39.580 epoch mà không đọc test hoặc fit thống kê tín hiệu; BCI pilot assignment cũ không đổi.
- 2026-08-15: Representative three-seed gates được khóa là PhysioNet←Sleep tại sequential forward stage 2→3 và Sleep←PhysioNet tại sequential reverse stage 1→2, kèm joint reference cùng seed. Mỗi task phải được đánh giá fidelity, continuous IG agreement và PED-vs-null trước khi scale checkpoint của task đó.
- 2026-08-17: External artifact audit xác nhận XAI v1 không reliable: raw PED ordering không được dùng làm method claim. Story chuyển thành reliability-aware physiological explanation drift; training/checkpoint/performance forgetting giữ nguyên. Chỉ XAI estimator được thay, bắt đầu bằng một BCI-Before-seed-42 margin/random-mask ridge pilot.
- 2026-08-17: Reliability v2 pilot xác nhận core idea nhưng bác bỏ additive surrogate: exact seed-42 v1 BA-drop cosine `0,3498`; single-cell margin-drop tăng lên `0,7668`; randomized-mask ridge coefficient cosine `0,9068` và per-subject `0,9214/0,9266`, nhưng held-out mask `R²=-1,4976/-0,9388`. Vì ridge không predict được unseen mask effect, gate chính thức là inconclusive và full scale không được phép. Candidate tiếp theo phải là single-cell margin-drop, không dùng ridge map làm explanation.
- 2026-08-17: Artifact audit xác nhận single-cell margin reliability đúng unit subject: A06 `0,9226` (66→61 positive cells), A07 `0,8040` (66→55); group cosine `0,7668` không dùng làm headline. Fixed-count randomized masks còn làm coefficient-sum không identifiable, nên ridge branch đóng vĩnh viễn.
- 2026-08-17: PhysioNet seed-42 single-cell margin gate fail: mọi cosine finite nhưng median per-subject chỉ `0,3578` so với gate `0,70`, và chỉ `3/18=0,1667` subject đạt `>=0,50` so với yêu cầu 2/3. Group cosine `0,6346` không override subject gate. Theo predeclared rule, dừng XAI; không chạy Sleep, PED v2 hoặc full method scale.
- 2026-08-17: Artifact audit định lượng được nguyên nhân PhysioNet fail và nó không phải estimator. Signal-to-noise của map là `4,31` ở BCI so với `1,49` ở PhysioNet, do PhysioNet chỉ có 90 row/subject (BCI ~490) và effect nhỏ hơn một nửa (max margin drop `0,144` so với `0,274`). Cắt BCI xuống đúng 44 row/subject/nửa cũng chỉ còn `0,722/0,541`, dưới gate. Forecast trước đó nói PhysioNet sẽ pass là sai vì đọc trục theo row/class thay vì row/nửa.
- 2026-08-17: Vì PhysioNet hết dữ liệu ở 90 row/subject nhưng Sleep có median `2.386` epoch/subject, quyết định mở đúng một pilot Sleep sample-size amendment v3 trước khi đóng XAI vĩnh viễn. Đây là thực thi tier Sleep đã có trong plan, không phải đổi gate sau khi thấy kết quả: gate subject-level giữ nguyên `median>=0,70` và `>=2/3 subject >=0,50`, ba outcome được khai báo trước, và cap 400 bị cấm vì lệch class composition. Một forward duy nhất ở cap 200; các mức nhỏ hơn tính offline nhờ tính lồng nhau của sha256 rank.
- 2026-08-17: Final Sleep amendment outcome: margin reliability pass rất mạnh ngay cap 20/50/100/200 với median subject cosine `0,9707/0,9898/0,9924/0,9967` và 15/15 subject `>=0,50` ở mọi cap. Vì vậy sample size không phải blocker cho Sleep. BCI margin fidelity pass (`BA drop 0,2506 > p95 0,1211`; margin drop `1,9042 > p95 0,7249`). Sleep exhaustive top-3 fidelity fail cả BA (`0,2999 < p95 0,3651`) và margin (`1,4805 < p95 2,0358`), cho thấy individually stable ranking không tạo thành faithful multi-cell set. Theo locked conjunction gate, XAI đóng; không PED/method scale.
- 2026-08-17: Sau dataset audit, cho phép một High-Gamma replacement gate thay vì nới metric trên PhysioNet. NEMAR manifest 33,6 GB chứa sourcedata+BIDS duplicate; chỉ tải BIDS cần thiết. Dataset selection, 22-channel montage, subject split và single-seed reliability gate được khóa trước download/model performance.
- 2026-08-18: `joint_high_gamma_v1` giữ đúng budget và hoàn tất ba seed. Mean test subject BA của offline reference: BCI `0,4449`, High-Gamma `0,6498`, Sleep `0,6594`; các số này chỉ xác nhận execution, không quyết định XAI gate.
- 2026-08-18: Joint strict all-six explanation gate fail. BCI pass 3/3 với median reliability `0,9154–0,9254` và cả margin/BA fidelity pass. High-Gamma reliability pass 3/3 (`0,9873–0,9902`) nhưng fidelity chỉ pass seed 2026; seeds 3407/42 fail cả margin (`1,3479<1,3996`, `1,4445<1,5809`) và BA (`0,2363<0,2439`, `0,2368<0,2521`). Theo predeclared rule, action là `discard_joint_hypothesis`; không tính `delta_A`, không sửa manuscript và không mở thêm joint experiment.
- 2026-08-18: Post-gate exploratory seeds 7/123/999 được khóa cùng lúc và High-Gamma pass 3/3: reliability medians `0,9871/0,9868/0,9835`; margin fidelity lần lượt `2,2993>1,4953`, `1,9346>1,3913`, `1,9152>1,4412`; BA `0,3589>0,2475`, `0,3027>0,2194`, `0,3000>0,2336`. Gộp formal+exploratory, High-Gamma pass 4/6 seed, xác nhận seed sensitivity nhưng không thỏa strict all-six và không được dùng để hồi sinh `delta_A`/manuscript claim. Result SHA-256: seed 7 `f68c20f8...ce9d2e`, 123 `f44085ae...cc51f`, 999 `7a672994...7555a`.
- 2026-08-18: Exploratory BCI joint maps cho seeds 7/123/999 cũng pass 3/3. Reliability medians `0,9210/0,9131/0,9227`; margin fidelity `1,8080>1,0303`, `1,0792>0,9658`, `2,2032>0,9615`; BA `0,2695>0,1875`, `0,1815>0,1593`, `0,2932>0,1711`. Vì vậy cả sáu exploratory task-seed maps pass, nhưng provenance vẫn là post-gate sensitivity và chưa có ensemble/`delta_A` analysis.
- 2026-08-17: High-Gamma metadata audit xác nhận 14 subject, 480–1.057 event/subject cân bằng 4 class và đủ toàn bộ BCI-22 channels. Real BDF sub-1 test contract pass sau khi lazy loader được sửa để pick 22/128 channels trước preload: 160 trial, 40/class, finite `22x4x200` trong CBraMod units. Full suite vẫn `113 passed`; signal download tiếp tục checksum-resumable.
- 2026-08-17: High-Gamma signal download hoàn tất 28/28 BDF và khớp source size. Source event audit phát hiện duy nhất `sub-2` train có class counts `202/204/204/203`; các run khác bằng nhau hoặc lệch tối đa một trial. Candidate protocol giữ toàn bộ published event, khóa `maximum_class_count_difference: 2` thay vì trim dữ liệu, và vẫn fail loud nếu thiếu class hoặc vượt ngưỡng quan sát này.
- 2026-08-17: Candidate cache build hoàn tất sau class-count correction: 6.510 train sample/7 subject và 3.934 validation sample/4 subject, không materialize hoặc đọc test split. Single-task seed 42 chạy đủ 2.500 step; final mean subject BA `0,6391`, best validation `0,6576` tại step 1.400. Final checkpoint SHA-256 `19616935...32a52c`; reliability gate phải dùng exact-final checkpoint theo protocol, không dùng best checkpoint.
- 2026-08-18: High-Gamma final candidate gate pass. Subject split-half cosine là `0,9838/0,9918/0,9928/0,9930`, median `0,9923`, group cosine `0,9857`; 4/4 subject vượt `0,50`. Conditional top-22 fidelity pass margin (`1,8010 > random p95 1,5944`) và BA (`0,25627 > random p95 0,25530`). BA margin chỉ hơn threshold khoảng `0,00097`, nên phải báo là pass sát và giữ full random distribution trong artifact. Result/config/individual/fidelity SHA-256 lần lượt là `a7298193...02637` / `32557787...4a125` / `91c1e849...0a4925` / `7e185fc6...24ac0`; test vẫn chưa được load trong gate.
- 2026-08-18: Sau khi v5 immutable, High-Gamma test cache mới được materialize: 3.040 sample trên subject 12–14. Replacement Sequential FT hoàn tất 9/9. Mean directed `F_rel`: `BCI←High-Gamma -0,2548` (n=3), `BCI←Sleep 0,0670` (n=6), `High-Gamma←BCI 0,1563` (n=6), `High-Gamma←Sleep 0,1528` (n=6), `Sleep←High-Gamma 0,6321` (n=3), `Sleep←BCI 0,1129` nhưng chỉ n=2 hợp lệ. Seed-42 reverse có Sleep trước BCI `0,2482`, headroom `0,0482 < 0,05`, nên raw forgetting `-0,05125` được giữ nhưng `F_rel=null` đúng rule. Stability summary đạt 4 signal và 5 sign-consistent directions nhưng `enough_replicates=false`, do đó overall gate fail và EWC/DER++ chưa được mở. Không thay threshold hoặc âm thầm dùng raw F sau khi thấy outcome.
- 2026-08-18: User-approved v7 summary pass mà không đổi v6 result: `Sleep←BCI` gate dùng raw F cho cả ba seed (mean `0,00058`, SD `0,04706`, không phải signal), năm direction khác giữ `F_rel`. Tổng cộng đủ replicate 6/6, signal 4/4 required và sign-consistent 6/4 required. Immutable v7 summary SHA-256 `bfd2bd9f...f6e23e`; EWC/DER++ được mở với hyperparameter cũ, không retune.
- 2026-08-18: Replacement EWC và DER++ hoàn tất 18/18 run; mọi stage/checkpoint/prediction digest được summary verifier đọc lại. Mean directed `F_rel` theo thứ tự Sequential/EWC/DER++: `BCI←High-Gamma -0,2548/-0,1126/-0,0695`; `BCI←Sleep 0,0670/0,0642/-0,0532`; `High-Gamma←BCI 0,1563/-0,0091/0,0358`; `High-Gamma←Sleep 0,1528/0,0044/0,0029`; `Sleep←High-Gamma 0,6321/0,0188/0,0592`. `Sleep←BCI` là raw-fallback cho Sequential, còn EWC/DER++ `F_rel=-0,0005/0,0521`. EWC/DER++ giảm mạnh forgetting của High-Gamma và Sleep; inherited signal gate false (3/6 và 2/6 signal direction) vì forgetting bị suppress, không phải execution failure. Summary SHA-256 Sequential-v7/EWC/DER++ là `bfd2bd9f...f6e23e` / `1539ab85...2fc1e` / `98107895...4f28b`.
- 2026-08-18: Replacement PED v8 hoàn tất 27/27 run, 63/63 immediate transition cell và 108 immutable map artifact. Mean subject-level noise-corrected PED theo Sequential/EWC/DER++: `BCI←High-Gamma 0,2431/0,0146/0,0516`; `BCI←Sleep 0,1400/0,0167/0,0687`; `High-Gamma←BCI 0,0825/0,00032/0,0333`; `High-Gamma←Sleep 0,1691/0,0097/0,0627`. EWC và DER++ giảm PED so với matched Sequential ở 21/21 cell; mean paired delta `-0,1369/-0,0922`. Performance–PED Spearman ở 21 cell/method là Sequential `-0,370`, EWC `0,088`, DER++ `-0,214` (exploratory, không significant), nên performance retention không phải proxy cho explanation retention. Summary SHA-256 `2005ef2c...7cc43`.
- 2026-08-18: Paper artifacts khóa từ v8 summary: Figure 1 protocol + all-three-seed BCI reliance maps, Figure 2 63-cell performance/PED scatter + 21-cell paired deltas, và Table 1 direction-level `F_rel/PED`. Order–seed-blocked inference cho mean PED delta EWC `-0,1222` (95% bootstrap CI `[-0,1481,-0,0955]`) và DER++ `-0,0804` (`[-0,1032,-0,0581]`); cả hai 9/9 block giảm, two-sided sign-test `p=0,00390625`. Full LaTeX draft, bibliography, source notes và validation report nằm dưới `paper/`. Tectonic 0.16.9 + bundled IEEEtran compile pass thành 4 trang US Letter: trang 1--3 technical, trang 4 references-only; không overfull/undefined citation. Chỉ còn author metadata và final official-kit/PDF-eXpress compliance.

## Validation

- Focused proof:
  - Unit tests cho rFFT decomposition/reconstruction, exact one-cell removal, untouched-cell identity và finite output trên 2/22-channel, 4/30-patch signals.
  - Subject-level BA-drop aggregation, positive-map normalization, JSD/rank-drift math, empty-positive-mass rejection và deterministic random masks.
  - Spectral-mask IG completeness/finite check trên synthetic additive signal và exact cell registry alignment với occlusion.
  - Config/checkpoint/cache/sample-ID digest tamper checks và immutable-output overwrite guard.
- Integration proof:
  - Pilot BCI-only/post-Sleep/joint checkpoints strict-load, reproduce stored unoccluded BCI metrics và generate maps trên frozen validation `attribution_fit/gate` IDs mà không đọc test.
  - Held-out top-20%-vs-100-random fidelity, IG rank agreement và same-checkpoint bootstrap/split null pass/fail được tạo bởi một digest-verified summary command.
  - Recompute final PED table/figures từ immutable result artifacts bằng một documented command.
- Statistical proof:
  - Subject-clustered uncertainty, paired method/order/seed/transition comparison, raw-`F` sensitivity và BCI fold robustness.
  - Không suy luận từ số trial/cell; report transition/seed coverage và distinguish exploratory association from attribution fidelity.
- Repository-required checks:
  - Plan progress và decisions được cập nhật sau mỗi gate.
  - Không overwrite manifest/config/result đã dùng trong paper.
  - Final manuscript numbers trace được về immutable run IDs.

Existing foundation proof completed on 2026-08-13/14:

- `python3.12 -m py_compile` passed for all source and test modules.
- `UV_CACHE_DIR=/tmp/eeg-forgetting-uv-cache uv run pytest -q`: 79 tests passed, including manifest/config tamper checks, main/reproduction cache round-trip, CBraMod identity/variable-shape/depth/train-mode checks, exact per-example Fisher extraction/math, sequential forgetting/near-chance/replicate-summary rules, EWC penalty math, byte-capped reservoir accounting/state, method-selection constraints, stage-resume RNG restoration, fixed early-stopping contract, overwrite guards, and a real MNE RawArray filter/resample/channel-order integration test.
- ZIP CRC passed for both BCI archives; PhysioNet and Sleep-EDF downloaded-file SHA-256 checks passed against official `SHA256SUMS.txt`.
- `scripts/audit_downloaded_data.py` observed 18 valid BCI GDF sessions, 109 x 6 PhysioNet imagery runs with explicit clean exclusions, and 153 paired Sleep recordings across 78 subjects.
- `scripts/audit_manifests.py --verify-sources` passed for active immutable manifest v3: BCI 5/2/2 subjects, PhysioNet clean 70/18/17 plus 4 explicit exclusions, and Sleep 48/15/15 subjects with 94/30/29 recordings. Every source, config, and manifest SHA-256 matched.
- Real preprocessing smoke through manifest v3 passed on A01E (281 non-artifact trials, 4 classes, `22x4x200`), S001R04 (15 trials, `22x4x200`), and SC00 night 1 (841 epochs, 5 stages, `2x30x200`); every signal was finite `float32` in CBraMod units.
- CBraMod adapter strict-loaded all 211 checkpoint tensors (4,924,000 parameters) and matched pinned upstream output exactly (`max_abs_diff=0.0`). On one L40S with PyTorch 2.13.0+cu130, real batch-size-2 forward/loss/backward passed for BCI, PhysioNet and Sleep with finite non-zero last-block gradients.
- PhysioNet reproduction cache matched the pinned upstream preprocessing exactly on S001R04 (`15x64x4x200`, `max_abs_diff=0.0`), contained the same 9.837 examples across 70/19/20 subjects, and reproduced the upstream classifier initialization bit-for-bit. The completed 50-epoch run selected epoch 33 and produced test BA 0,6229, kappa 0,4972 and weighted-F1 0,6241; checkpoint/config digests re-verified after the run.
- Frozen test caches were materialized and bound to manifest v3: BCI 1.036 samples/2 subjects, PhysioNet 1.530/17 and Sleep 37.227/15. Full source-checksum manifest audit passed immediately before launching sequential FT.
- Sequential FT completed 9/9 immutable runs (three orders × three seeds). The predeclared stability gate passed with six signal-bearing and six sign-consistent directed transitions; the summary binds every input result by SHA-256.
- EWC and DER++ each completed 9/9 immutable runs. Recomputed summaries verified every stage/checkpoint/prediction digest. Peak persistent state was 25,766,400 bytes for two offline-EWC Fisher/anchor states and 8,381,067 bytes for DER++'s 119-slot reservoir.
- XAI implementation validation: locked-config dry run strict-loaded all three pilot checkpoints and verified 982 validation rows, frozen 489/493 split, 110-cell registry and all cache/checkpoint/config digests. Full repository suite passed `97 passed`; focused proof covers continuous-trial rFFT reconstruction, exact band removal, NumPy frozen-index handling, subject-equal reliance, base-2 JSD, spectral-mask IG completeness and legacy-summary compatibility.
- RQ2 BCI←Sleep scale preflight strict-loaded before/after checkpoints từ Sequential-forward-3407, EWC-challenging-42 và DER++-forward-2026, đồng thời verified parent config/summary/result/stage/checkpoint/cache/split digests. Full repository suite passed `99 passed` trước launch.
- PhysioNet/Sleep attribution preflight strict-loaded sequential before/after và joint checkpoint, verified capped assignment SHA cùng validation/test-cache SHA, và full repository suite passed `102 passed` trước launch.
- Reliability v2 preflight strict-loaded exact BCI-Before seed-42 checkpoint, verified frozen split/test-blind digests, passed synthetic continuous-margin/ridge recovery proof và full suite `106 passed`. Immutable result SHA-256 `946a7044...30d17bc`; margin-score artifact SHA-256 `171a0ac1...0fb35dc`.
- High-Gamma source verification và cache build pass cho đủ 28 BDF: 6.510 train + 3.934 validation sample; test subjects 12–14 không được materialize. Sau class-count regression tests, full repository suite pass `115 passed`. Candidate seed-42 training hoàn tất 2.500/2.500 step và ghi immutable `best.pt`, `final.pt`, `result.json` với cache/config/checkpoint digests.
- High-Gamma attribution preflight strict-loaded exact-final single-task checkpoint, verified 3.934 validation row, frozen 1.966/1.968 split và 110-cell registry; reliability/fidelity result pass và bind config/checkpoint/score artifacts bằng SHA-256. Full suite sau configurable task registry, replay registry và stability-amendment tests pass `120 passed`.
- Replacement performance matrix hoàn tất 27/27 run: Sequential/EWC/DER++ cùng 3 order × 3 seed. Summary verifier rehashed toàn bộ stage JSON, checkpoint và prediction; EWC peak state giữ 25.766.400 byte, DER++ peak 8.381.067 byte/119 slot.
- Replacement PED scale hoàn tất 27/27 XAI result và 108 map artifact; summary verifier rehashed mọi artifact, xác nhận đúng 63 transition cell và `test_was_loaded_for_xai=false`. Noise-corrected cross-minus-within JSD unit tests và full repository suite pass `122 passed`.
- Paper artifact builder tái tạo 2 figures ở PDF/SVG/PNG, Table 1 và deterministic 20.000-replicate clustered statistics. PNG final-aspect visual inspection pass; manuscript validator xác nhận đủ artifacts/citations, source summary SHA, 4 PDF pages và references-only final page, status `draft_ready_author_metadata_pending`. Full suite tiếp tục pass `122 passed`.

## Result

Foundation, full CL matrices và performance forgetting vẫn hợp lệ; Fisher headline và XAI raw-PED claims bị rút lại. Margin-drop reliable trên hai BCI validation subjects nhưng fail cross-subject PhysioNet gate (`median=0,3578`, `3/18 >=0,50`).

Nguyên nhân fail đã được định lượng là trial/subject chứ không phải estimator: signal-to-noise `4,31` (BCI, ~490 row/subject) so với `1,49` (PhysioNet, 90 row/subject), và BCI cắt xuống ngân sách PhysioNet cũng chỉ còn `0,722/0,541`. PhysioNet không còn dữ liệu để lấy thêm, nhưng Sleep-EDF có median `2.386` epoch/subject nên ràng buộc này là tự đặt qua cap 20.

Final Sleep exception fail fidelity, nhưng High-Gamma replacement candidate đã pass reliability/fidelity và hoàn tất full performance + PED scale. Sequential có subject-mean PED lớn dù performance đôi khi tăng, rõ nhất `BCI←High-Gamma`: `F_rel=-0,255` nhưng PED `0,243`. EWC giảm PED xuống `0,0003–0,0167` theo direction và DER++ xuống `0,0333–0,0687`; cả hai thắng matched Sequential 21/21 cell. Within-method performance–PED correlation vẫn yếu. Joint offline-reference hypothesis đã bị loại vì strict all-six gate fail trên 2/3 High-Gamma joint seeds; không tính `delta_A` và manuscript giữ nguyên. Còn lại là author metadata/internal review/submission validation.
