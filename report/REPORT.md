# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  {
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },

- Record này mô tả toàn ảnh như thế nào?
	+ Checkpoint này sử dụng danh mục nhãn (taxonomy) ImageNet-1K và xếp hạng các lớp mô tả cho toàn bộ bức ảnh; mô hình không xác định vị trí từng vật thể hay phân đoạn từng điểm ảnh.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
	+ Danh sách lớp (class list) được định nghĩa trong file cấu hình của checkpoint (ví dụ: `data.yaml` hoặc file config tương ứng) và thường dựa trên các tập dữ liệu huấn luyện lớn như ImageNet, COCO hoặc tập dữ liệu chuyên dụng.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
	+ class_id: Là định danh số duy nhất cho mỗi lớp trong mô hình 
	+ class_name: Là nhãn con người có thể đọc được; 
	+ taxonomy_name: Chỉ rõ tập dữ liệu/hệ thống phân loại mà mô hình được huấn luyện, giúp phân biệt giữa các phiên bản hoặc các tập dữ liệu khác nhau.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
	+ Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ liệu mô hình có phân loại tất cả các chủ thể hay chỉ chủ thể nổi bật nhất, cách xếp hạng các chủ thể phụ và định nghĩa ngưỡng tin cậy (confidence threshold) để quyết định khi nào ghi nhận một phân loại.

- Vì sao model score không phải ground truth?
	+ Model score là dự đoán của mô hình, không phải nhãn đúng của ảnh
	+ Ground truth là nhãn đúng của ảnh, được gán thủ công bởi con người


## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
{
    "class_name": "person",
    "score": 0.769676,
    "bbox_xyxy": [
      0.08,
      256.79,
      18.39,
      313.12
    ],
    "bbox_width": 18.32,
    "bbox_height": 56.33
  },

- Diễn giải vị trí box bằng lời:
    + Mỗi prediction mô tả một vật thể phát hiện được kèm theo tên lớp, điểm số và tọa độ bounding box bbox_xyxy = [x_min, y_min, x_max, y_max], quy ước gốc tọa độ (0, 0) nằm ở góc trên bên trái ảnh.

- So sánh số prediction ở hai threshold:
    + Ngưỡng lọc DETECTION_SCORE_THRESHOLD chỉ dùng để lọc prediction của mô hình. Hạ thấp ngưỡng sẽ giữ lại nhiều prediction hơn nhưng dễ lẫn kết quả sai; tăng ngưỡng giúp kết quả chắc chắn hơn nhưng dễ bỏ sót vật thể.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    + Độ bao phủ: Ngưỡng thấp giúp tăng độ bao phủ, hạn chế bỏ sót vật thể khó/nhỏ nhưng làm tăng dự đoán sai. Ngưỡng cao giúp các dự đoán chính xác hơn nhưng giảm độ bao phủ và dễ bỏ sót đối tượng. 
    + Khối lượng công việc của reviewer: Ngưỡng thấp làm tăng số lượng prediction, tăng khối lượng công việc xem xét và gán nhãn thủ công. Ngưỡng cao giảm số lượng, giảm workload nhưng có thể tăng số lượng false negatives.

- Đề xuất một quy tắc box chặt:
    + Khung box phải bao sát 4 điểm cực trị (trên cùng, dưới cùng, mép trái, mép phải) của phần nhìn thấy của vật thể; khoảng cách lề thừa không vượt quá 2–3 pixel và tuyệt đối không cắt lẹm vào ranh giới của vật thể.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    + Nếu vật thể bị che khuất hơn 50% hoặc bị cắt ra khỏi khung hình mà không thể xác định toàn bộ hình dáng, annotator nên: (1) Ưu tiên giữ lại box nếu vật thể dễ nhận dạng; (2) Thu hẹp box sát mép ảnh và ghi chú đặc biệt; (3) Loại bỏ nếu không thể chắc chắn về danh tính (để tránh rác);

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
{
    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],
      [
        458.0,
        71.0
      ],
      [
        456.0,
        71.0
      ],
      [
        455.0,
        70.0
      ]
    ]
  },


- Polygon bổ sung chi tiết gì so với box?
    + Khác với phát hiện vật thể, mỗi cá thể ở đây có thêm trường polygon_xy: danh sách các cặp tọa độ [x, y] mô tả chính xác đường biên bao quanh vật thể.

- `instance_id` dùng để làm gì và không phải loại ID nào?
    + Instance_id: Mã định danh instance_id dùng để phân biệt các cá thể riêng biệt trong ảnh, không phải là class ID hay tracking ID.

- Đề xuất một quy tắc biên mask:
    + Cần quy định rõ ràng về việc đóng gói từng object thành một instance riêng biệt, cách đánh số instance_id, ngưỡng tối thiểu số điểm (score_threshold) và cách xử lý các vật thể bị che khuất hoặc tiếp xúc. 
    
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    + Nếu vùng mờ che khuất hơn 50% vật thể → loại bỏ hoặc đánh dấu đặc biệt;
	+ Nếu vật thể tiếp xúc/dính vào nhau → đánh dấu riêng lẻ nếu có thể tách biệt, nếu không thể tách biệt thì gộp chung hoặc loại bỏ.


## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  gán nhãn đơn: class_id, class_name, taxonomy | - Ảnh có nhiều chủ thể khác nhau. - Chủ thể quá nhỏ hoặc mờ. | - Đọc guideline để xác định chủ thể. - Chọn 1 nhãn đúng nhất trong danh mục. | - Kiểm tra nhãn gán có đúng với chủ thể chính theo guideline không. |
| Phát hiện vật thể | gán class_name, class_id, kèm tọa độ bounding box bbox_xyxy | - Bỏ sót các vật thể nhỏ/ở xa/bị che khuất. - Nhầm lẫn một đối tượng bị che thành 2 vật thể riêng biệt. | - Gán đúng nhãn cho từng box. - Đánh dấu cờ is_occluded/is_truncated theo guideline.| - Rà soát các vật thể bị bỏ sót hoặc vẽ thừa. - Kiểm tra tính chính xác của nhãn lớp. |
| Instance segmentation |  gán instance_id, class_name, và chuỗi đỉnh đa giác polygon_xy | - Đường biên thô ráp, ít điểm. - Gộp nhầm 2 cá thể đứng sát nhau thành 1 mask. - Mask lấn sang nền/bóng đổ hoặc vật thể khác. | - Vẽ đường viền polygon ôm sát ranh giới pixel của từng cá thể. - Đặt instance_id riêng biệt cho mỗi đối tượng. - Xử lý tách riêng các đối tượng dính nhau. | - Soát độ chính xác của đường biên mask. - Đảm bảo không bị dính mask giữa các cá thể.|

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    + Không đồng nhất điểm số tin cậy (score/confidence) của mô hình với nhãn chuẩn (ground truth).
    + Không để lộ họ tên, MSSV hoặc dữ liệu cá nhân trong báo cáo, output hay file ZIP.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    + Lab Coach thông qua kênh hỏi đáp (Slack/Discord/zalo/email).

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
